/**
 * \file
 * \brief Routines for linear interpolation of vertical atmospheric profiles
 * \author
 *   Kelton Halbert                  \n
 *   Email: kelton.halbert@noaa.gov  \n
 * \date   2022-10-13
 *
 * Written for the NWS Storm Predidiction Center \n
 * Based on NSHARP routines originally written by
 * John Hart and Rich Thompson at SPC.
 */

#ifndef SHARP_INTERP_H
#define SHARP_INTERP_H

#include <SHARPlib/algorithms.h>
#include <SHARPlib/constants.h>
#include <SHARPlib/qc.h>

#include <cmath>
#include <cstddef>
#include <functional>

namespace sharp {

#ifndef NO_QC
template <typename CoordArr, typename DataArr>
SHARP_HD inline bool widen_bracket_past_missing(const float coord_val,
                                                const CoordArr coord_arr,
                                                const DataArr data_arr,
                                                const std::ptrdiff_t N,
                                                std::ptrdiff_t& idx_bot,
                                                std::ptrdiff_t& idx_top,
                                                float& result) {
    const bool bot_missing = is_missing(data_arr[idx_bot]);
    const bool top_missing = is_missing(data_arr[idx_top]);
    if (!bot_missing && !top_missing) return false;

    if (!bot_missing && (coord_val == coord_arr[idx_bot])) {
        result = data_arr[idx_bot];
        return true;
    }
    if (!top_missing && (coord_val == coord_arr[idx_top])) {
        result = data_arr[idx_top];
        return true;
    }

    for (; idx_bot > 0; --idx_bot) {
        if (!is_missing(data_arr[idx_bot])) break;
    }

    for (; idx_top < N - 1; ++idx_top) {
        if (!is_missing(data_arr[idx_top])) break;
    }

    // in the case the data are still missing at this point,
    // return a missing value
    if (is_missing(data_arr[idx_bot]) || is_missing(data_arr[idx_top])) {
        result = MISSING;
        return true;
    }
    return false;
}
#endif

/**
 * \author C++20 Standard Template Library
 *
 * \brief Linearly interpolates between a and b with distance t.
 *
 * This routine was copied from the C++20 standard template library.
 * It is used to linearly interpolate between A and B over a normalized
 * distance T, where 0 <= T <= 1. Full documentation and details can be found
 * in the paper located here:
 * https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p0811r3.html
 *
 * This linear interpolation guarantees the following:
 *
 * 1. Exactness: lerp(a,b,0)==a & lerp(a,b,1)==b
 * 2. Monotonicity: cmp(lerp(a,b,t2), lerp(a,b,t1)) * cmp(t2, t1) <!--
 * -->* cmp(b,a) >= 0, where cmp is an arithmetic three-way comparison function
 * 3. Determinacy: result of NaN only for lerp(a,a,INFINITY)
 * 4. Boundedness: t<0 || t>1 || isfinite(lerp(a,b,t))
 * 5. Consistency: lerp(a,a,t) == a
 *
 * \param   __a     left value
 * \param   __b     right value
 * \param   __t     where 0 <= __t <= 1.0
 *
 * \return  The value between __a and __b at distance __t between them.
 */
template <typename _Fp>
[[nodiscard]] SHARP_HD constexpr _Fp __lerp(const _Fp __a, const _Fp __b,
                                            const _Fp __t) {
    if ((__a <= 0 && __b >= 0) || (__a >= 0 && __b <= 0))
        return __t * __b + (1 - __t) * __a;

    if (__t == 1) return __b;  // exact

    // Exact at __t=0, monotonic except near __t=1,
    // bounded, determinate, and consistent:
    const _Fp __x = __a + __t * (__b - __a);
    return (__t > 1) == (__b > __a)
               ? (__b < __x ? __x : __b)
               : (__b > __x ? __x : __b);  // monotonic near __t=1
}

/**
 * \author C++20 Standard Template Library
 *
 * \brief Linearly interpolates between a and b with distance t.
 *
 * This routine was copied from the C++20 standard template library.
 * It is used to linearly interpolate between A and B over a normalized
 * distance T, where 0 <= T <= 1. Full documentation and details can be found
 * in the paper located here:
 * https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p0811r3.html
 *
 * This linear interpolation guarantees the following:
 *
 * 1. Exactness: lerp(a,b,0)==a & lerp(a,b,1)==b
 * 2. Monotonicity: cmp(lerp(a,b,t2), lerp(a,b,t1)) * cmp(t2, t1) *
 * cmp(b,a) >= 0, where cmp is an arithmetic three-way comparison function
 * 3. Determinacy: result of NaN only for lerp(a,a,INFINITY)
 * 4. Boundedness: t<0 || t>1 || isfinite(lerp(a,b,t))
 * 5. Consistency: lerp(a,a,t) == a
 *
 * \param   __a     left value
 * \param   __b     right value
 * \param   __t     where 0 <= __t <= 1.0
 *
 * \return  The value between __a and __b at distance __t between them.
 */
[[nodiscard]] SHARP_HD constexpr float lerp(const float __a, const float __b,
                                            const float __t) {
    return __lerp(__a, __b, __t);
}

/**
 * \author Kelton Halbert - NWS Storm Prediction Center
 *
 * \brief Linearly interpolate a data field to a requested height level.
 *
 * Interpolates data in height coordinates (meters) when given the
 * array of heights, the array of data to interpolate, a height
 * level to interpolate to, and the number of levels in the arrays.
 *
 * The height array must be sorted in ascending order and monotonic.
 * For performance reasons, this routine assumes your data is well
 * ordered. If weird stuff happens, check the ordering of your array
 * values first.Duplicate height values or decreasing height values
 * may produce unexpected results.
 *
 * By default, this routine skips levels whose data is sharp::MISSING or
 * NaN. A height_val exactly on a valid level returns that level's value.
 * Otherwise, it interpolates between the nearest valid levels above and
 * below height_val, and returns sharp::MISSING if either side has none.
 * Building with NO_QC turns these checks off. An empty profile (N < 1)
 * always returns sharp::MISSING.
 *
 * \param   height_val  The height value to interpolate data to
 * \param   height_arr  The array of height values in the profile
 * \param   data_arr    The array of data values you want interpolated
 * \param   N           The the array length (number of elements)
 *
 * \return  The value of data_arr at the requested height_val.
 */
template <typename Ptr1 = const float*, typename Ptr2 = const float*>
[[nodiscard]] SHARP_HD inline float interp_height(const float height_val,
                                                  const Ptr1 height_arr,
                                                  const Ptr2 data_arr,
                                                  const std::ptrdiff_t N) {
    if (N < 1) return MISSING;
#ifndef NO_QC
    if (is_missing(height_val)) return MISSING;
#endif
    // If the height value is beyond the top of the profile,
    // or below the surface, we can't reasonably extrapolate
    if ((height_val > height_arr[N - 1]) || (height_val < height_arr[0]))
        return MISSING;

    if (N == 1) {
#ifndef NO_QC
        if (is_missing(data_arr[0])) return MISSING;
#endif
        return data_arr[0];
    }

    constexpr auto comp = std::less<float>();
    std::ptrdiff_t idx_top = upper_bound(height_arr, N, height_val, comp);
    std::ptrdiff_t idx_bot = idx_top - 1;

#ifndef NO_QC
    float bridged = MISSING;
    if (widen_bracket_past_missing(height_val, height_arr, data_arr, N, idx_bot,
                                   idx_top, bridged))
        return bridged;
#endif

    const float height_bot = height_arr[idx_bot];
    const float height_top = height_arr[idx_top];
    const float data_bot = data_arr[idx_bot];
    const float data_top = data_arr[idx_top];

    // normalize the distance between values
    // to a range of 0-1 for the lerp routine
    const float dz_norm = (height_val - height_bot) / (height_top - height_bot);

    // return the linear interpolation
    return lerp(data_bot, data_top, dz_norm);
}

[[nodiscard]] SHARP_HD inline float interp_height(const float /*height_val*/,
                                                  std::nullptr_t,
                                                  std::nullptr_t,
                                                  const std::ptrdiff_t /*N*/) {
    return MISSING;
}

/**
 * \author Kelton Halbert - NWS Storm Prediction Center
 *
 * \brief Linearly interpolate a data field to a requested pressure level.
 *
 * Interpolates data in pressure coordinates (mb) when given the
 * array of pressure values, the array of data to interpolate, a
 * pressure level to interpolate to, and the number of levels in
 * the arrays.
 *
 * The pressure array must be sorted in descending order and monotonic.
 * Duplicate pressure values or increasing pressure values may produce
 * unexpected results.
 *
 * Handles missing data and empty profiles the same way as
 * sharp::interp_height.
 *
 * \param   pressure_val    The pressure value to interpolate data to
 * \param   pressure_arr    The array of pressure values in the profile
 * \param   data_arr        The array of data values you want interpolated
 * \param   N               The the array length (number of elements)
 *
 * \return  The value of data_arr at the requested pressure_val.
 */
template <typename Ptr1 = const float*, typename Ptr2 = const float*>
[[nodiscard]] SHARP_HD inline float interp_pressure(const float pressure_val,
                                                    const Ptr1 pressure_arr,
                                                    const Ptr2 data_arr,
                                                    const std::ptrdiff_t N) {
    if (N < 1) return MISSING;
#ifndef NO_QC
    if (is_missing(pressure_val)) return MISSING;
#endif
    // If the pressure value is beyond the top of the profile,
    // or below the surface, we can't reasonably extrapolate
    if ((pressure_val < pressure_arr[N - 1]) ||
        (pressure_val > pressure_arr[0])) {
        return MISSING;
    }

    if (N == 1) {
#ifndef NO_QC
        if (is_missing(data_arr[0])) return MISSING;
#endif
        return data_arr[0];
    }

    constexpr auto comp = std::greater<float>();
    std::ptrdiff_t idx_top = upper_bound(pressure_arr, N, pressure_val, comp);
    std::ptrdiff_t idx_bot = idx_top - 1;

#ifndef NO_QC
    float bridged = MISSING;
    if (widen_bracket_past_missing(pressure_val, pressure_arr, data_arr, N,
                                   idx_bot, idx_top, bridged))
        return bridged;
#endif

    const float pressure_bot = pressure_arr[idx_bot];
    const float pressure_top = pressure_arr[idx_top];
    const float data_bot = data_arr[idx_bot];
    const float data_top = data_arr[idx_top];

    // In order to linearly interpolate pressure properly, distance needs
    // to be calculated in log10(pressure) coordinates and normalized
    // between 0 and 1 for the lerp routine.
    const float dp_norm =
        (std::log10(pressure_bot) - std::log10(pressure_val)) /
        (std::log10(pressure_bot) - std::log10(pressure_top));

    // return the linear interpolation
    return lerp(data_bot, data_top, dp_norm);
}

[[nodiscard]] SHARP_HD inline float interp_pressure(const float /*pressure_val*/,
                                                    std::nullptr_t,
                                                    std::nullptr_t,
                                                    const std::ptrdiff_t /*N*/) {
    return MISSING;
}

/**
 * \author Kelton Halbert - NWS Storm Prediction Center
 *
 * \brief Find the pressure level of first occurrence of a given value
 *
 * Conduct a bottom-up search for the first occurrence of a given value,
 * and interpolate in order to get the pressure level it occurs at.
 *
 * An exact match returns that level's pressure. By default, the search
 * skips levels whose data is sharp::MISSING or NaN, so it can find a
 * crossing across a gap in the data. A sharp::MISSING or NaN data_val
 * returns sharp::MISSING. Building with NO_QC turns these checks off.
 *
 * \param   data_val        The value being searched for
 * \param   pressure_arr    The pressure array to get the level from (Pa)
 * \param   data_arr        The data array of values being searched over
 * \param   N               The length of the arrays (number of values)
 *
 * \return  pressure_level  (Pa)
 */
template <typename Ptr1 = const float*, typename Ptr2 = const float*>
[[nodiscard]] SHARP_HD inline float find_first_pressure(const float data_val,
                                                        const Ptr1 pressure_arr,
                                                        const Ptr2 data_arr,
                                                        const std::ptrdiff_t N) {
    std::ptrdiff_t k_start = 0;
#ifndef NO_QC
    if (is_missing(data_val)) {
        return MISSING;
    }
    for (; k_start < N; ++k_start) {
        if (!is_missing(data_arr[k_start])) break;
    }
#endif
    if ((k_start < N) && (data_arr[k_start] == data_val))
        return pressure_arr[k_start];

    for (std::ptrdiff_t k = k_start + 1; k < N; ++k) {
        float val0 = data_arr[k_start];
        float val1 = data_arr[k];
#ifndef NO_QC
        if (is_missing(val1)) continue;
#endif
        if (val1 == data_val) return pressure_arr[k];

        if ((data_val - val0) * (data_val - val1) < 0) {
            const float logp_bot = std::log10(pressure_arr[k_start]);
            const float logp_top = std::log10(pressure_arr[k]);

            const float d_norm = (data_val - val0) / (val1 - val0);

            return std::pow(10.0f, lerp(logp_bot, logp_top, d_norm));
        }

        k_start = k;
    }

    return MISSING;
}

/**
 * \author Kelton Halbert - NWS Storm Prediction Center
 *
 * \brief Find the height level of first occurrence of a given value
 *
 * Conduct a bottom-up search for the first occurrence of a given value,
 * and interpolate in order to get the pressure level it occurs at.
 *
 * Handles missing data the same way as sharp::find_first_pressure.
 *
 * \param   data_val    The value being searched for
 * \param   height_arr  The height array to get the level from (meters)
 * \param   data_arr    The data array of values being searched over
 * \param   N           The length of the arrays (number of values)
 *
 * \return  height_level    (meters)
 */
template <typename Ptr1 = const float*, typename Ptr2 = const float*>
[[nodiscard]] SHARP_HD inline float find_first_height(const float data_val,
                                                      const Ptr1 height_arr,
                                                      const Ptr2 data_arr,
                                                      const std::ptrdiff_t N) {
    std::ptrdiff_t k_start = 0;
#ifndef NO_QC
    if (is_missing(data_val)) {
        return MISSING;
    }
    for (; k_start < N; ++k_start) {
        if (!is_missing(data_arr[k_start])) break;
    }
#endif
    if ((k_start < N) && (data_arr[k_start] == data_val))
        return height_arr[k_start];

    for (std::ptrdiff_t k = k_start + 1; k < N; ++k) {
        float val0 = data_arr[k_start];
        float val1 = data_arr[k];
#ifndef NO_QC
        if (is_missing(val1)) continue;
#endif
        if (val1 == data_val) return height_arr[k];

        // will have a negative sign if levels straddle
        // the point being searched for
        if ((data_val - val0) * (data_val - val1) < 0) {
            const float hght_bot = height_arr[k_start];
            const float hght_top = height_arr[k];

            const float d_norm = (data_val - val0) / (val1 - val0);

            return lerp(hght_bot, hght_top, d_norm);
        }

        k_start = k;
    }

    return MISSING;
}

}  // end namespace sharp

#endif  // SHARP_INTERP_H
