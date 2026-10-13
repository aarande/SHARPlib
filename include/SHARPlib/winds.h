/**
 * \file
 * \brief Routines used to compute kinematic attributes and indices<!--
 * --> of vertical sounding profiles
 * \author
 *   Kelton Halbert                  \n
 *   Email: kelton.halbert@noaa.gov  \n
 * \date   2022-10-13
 *
 * Written for the NWS Storm Predidiction Center \n
 * Based on NSHARP routines originally written by
 * John Hart and Rich Thompson at SPC.
 */
#ifndef SHARP_WINDS_H
#define SHARP_WINDS_H

#include <SHARPlib/algorithms.h>
#include <SHARPlib/constants.h>
#include <SHARPlib/interp.h>
#include <SHARPlib/layer.h>

#include <cmath>
#include <cstddef>

namespace sharp {

/**
 * \author Kelton Halbert - NWS Storm Prediction Center
 *
 * \brief A simple structure of two named floats that represent a wind vector.
 */
struct WindVector {
    /**
     * \brief The magntidue of a wind vector
     */
    float speed;

    /**
     * \brief The direction (degrees from North) of a wind vector.
     */
    float direction;

    SHARP_HD constexpr WindVector() : speed(0.0f), direction(0.0f) {}
    SHARP_HD constexpr WindVector(float spd, float dir) : speed(spd), direction(dir) {}
};

/**
 * \author Kelton Halbert - NWS Storm Prediction Center
 *
 * \brief A simple structure of two names floats that represent<!--
 * --> the components of a wind vector.
 */
struct WindComponents {
    /**
     * \brief The zonal (U) component of a wind vector.
     */
    float u;
    /**
     * \brief The meridional (V) component of a wind vector.
     */
    float v;

    SHARP_HD constexpr WindComponents() : u(0.0f), v(0.0f) {}
    SHARP_HD constexpr WindComponents(float u_comp, float v_comp) : u(u_comp), v(v_comp) {}
};

/**
 * \author John Hart - NSSFC KCMO / NWSSPC OUN
 *
 * \brief Computes the zonal (U) wind component from a wind vector.
 *
 * Given the wind speed and direction, compute and return the zonal
 * (U) wind component.
 *
 * \param   wind_speed      (m/s)
 * \param   wind_direction  (degrees from North)
 *
 * \return  u_component     (m/s)
 */
[[nodiscard]] SHARP_HD SHARP_FORCEINLINE float u_component(float wind_speed, float wind_direction) {
#ifndef NO_QC
    if ((wind_direction == MISSING) || (wind_speed == MISSING)) return MISSING;
#endif
    wind_direction *= (PI / 180.0f);
    return -1.0f * wind_speed * std::sin(wind_direction);
}

/**
 *
 * \author John Hart - NSSFC KCMO / NWSSPC OUN
 *
 * \brief Computes the meridional (V) wind component from a wind vector.
 *
 * Given the wind speed and direction, compute and return the meridional
 * (V) wind component.
 *
 * \param   wind_speed      (m/s)
 * \param   wind_direction  (degrees from North)
 *
 * \return  v_component     (m/s)
 */
[[nodiscard]] SHARP_HD SHARP_FORCEINLINE float v_component(float wind_speed, float wind_direction) {
#ifndef NO_QC
    if ((wind_direction == MISSING) || (wind_speed == MISSING)) return MISSING;
#endif
    wind_direction *= (PI / 180.0f);
    return -1.0f * wind_speed * std::cos(wind_direction);
}

/**
 *
 * \author John Hart - NSSFC KCMO / NWSSPC OUN
 *
 * \brief Computes the direction (degrees from North) a vector points to.
 *
 * Given the zonal (U) and meridional (V) components of a vector, compute
 * and return the direction (degrees from North) the vector points to.
 *
 * \param   u_comp          (m/s)
 * \param   v_comp          (m/s)
 *
 * \return  wind_direction  (degrees from North)
 */
[[nodiscard]] SHARP_HD SHARP_FORCEINLINE float vector_angle(float u_comp, float v_comp) {
#ifndef NO_QC
    if ((u_comp == MISSING) || (v_comp == MISSING)) return MISSING;
#endif
    if ((u_comp == 0.0f) && (v_comp == 0.0f)) return 0.0f;

    float wind_direction =
        std::atan2(-1.0f * u_comp, -1.0f * v_comp) * (180.0f / PI);
    if (wind_direction < 0.0f) wind_direction += 360.0f;
    if (wind_direction < TOL) wind_direction = 0.0f;
    return wind_direction;
}

/**
 *
 * \author John Hart - NSSFC KCMO / NWSSPC OUN
 *
 * \brief Computes the magnitude of a vector given its components.
 *
 * Given the zonal (U) and meridional (V) components of a vector, compute
 * and return the magnitude (m/s) of the vector.
 *
 * \param   u_comp      (m/s)
 * \param   v_comp      (m/s)
 *
 * \return  wind_speed  (m/s)
 */
[[nodiscard]] SHARP_HD SHARP_FORCEINLINE float vector_magnitude(float u_comp, float v_comp) {
#ifndef NO_QC
    if ((u_comp == MISSING) || (v_comp == MISSING)) return MISSING;
#endif
    return std::sqrt((u_comp * u_comp) + (v_comp * v_comp));
}

/**
 *
 * \author John Hart - NSSFC KCMO / NWSSPC OUN
 *
 * \brief Precisely computes the magnitude of a vector given its components.
 *
 * Given the zonal (U) and meridional (V) components of a vector, compute
 * and return the magnitude (distance / time) of the vector. Instead of using
 * a standard sqrt(u*u + v*v) approach, this uses the c++ Standard Template
 * Library implementation of std::hypot. Using std::hypot is ~20 times slower,
 * but ensures a higher accuracy and checks for overflow/underflow when using
 * extremely large or extremely small values. Chances are, you are fine using
 * the imprecise version as long as values are in the range of
 * 1e-100 < values < 1e100.
 *
 * \param   u_comp      (m/s)
 * \param   v_comp      (m/s)
 *
 * \return  wind_speed  (m/s)
 */
[[nodiscard]] SHARP_HD SHARP_FORCEINLINE float vector_magnitude_precise(float u_comp, float v_comp) {
#ifndef NO_QC
    if ((u_comp == MISSING) || (v_comp == MISSING)) return MISSING;
#endif
    return std::hypot(u_comp, v_comp);
}

/**
 *
 * \author Kelton Halbert - NWS Storm Prediction Center
 *
 * \brief Conveniance function to compute wind speed and direction<!--
 * --> from components
 *
 * Given the zonal (U) and meridional (V) components of a vector, compute
 * and return the wind speed (m/s) and direction (degrees from
 * North) from the components as a struct.
 *
 * \param   u_comp  (m/s)
 * \param   v_comp  (m/s)
 *
 * \return  {wind_speed, wind_direction}
 */
[[nodiscard]] SHARP_HD SHARP_FORCEINLINE WindVector components_to_vector(float u_comp, float v_comp) {
    const float wind_speed = vector_magnitude(u_comp, v_comp);
    const float wind_direction = vector_angle(u_comp, v_comp);

    return {wind_speed, wind_direction};
}

/**
 *
 * \author Kelton Halbert - NWS Storm Prediction Center
 *
 * \brief Conveniance function to compute wind speed and direction<!--
 * --> from components
 *
 * Given the components of a vector via sharp::WindComponents, compute
 * and return the wind speed (m/s) and direction (degrees from
 * North) and return them as sharp::WindVector.
 *
 * \param   comp    {u_comp, v_comp}
 *
 * \return  {wind_speed, wind_direction}
 */
[[nodiscard]] SHARP_HD SHARP_FORCEINLINE WindVector components_to_vector(WindComponents comp) {
    const float wind_speed = vector_magnitude(comp.u, comp.v);
    const float wind_direction = vector_angle(comp.u, comp.v);

    return {wind_speed, wind_direction};
}

/**
 *
 * \author Kelton Halbert - NWS Storm Prediction Center
 *
 * \brief Conveniance function to compute the U and V components<!--
 * --> of a wind vector.
 *
 * Given the wind speed and direction of a vector, compute and return the
 * zonal and meridional vector components as a struct.
 *
 * \param   wind_speed      (m/s)
 * \param   wind_direction  (degrees from North)
 *
 * \return  {u_comp, v_comp}
 */
[[nodiscard]] SHARP_HD SHARP_FORCEINLINE WindComponents vector_to_components(float wind_speed,
                                                                             float wind_direction) {
#ifndef NO_QC
    if ((wind_direction == MISSING) || (wind_speed == MISSING))
        return {MISSING, MISSING};
#endif
    wind_direction *= (PI / 180.0f);
    const float u_comp = -1.0f * wind_speed * std::sin(wind_direction);
    const float v_comp = -1.0f * wind_speed * std::cos(wind_direction);

    return {u_comp, v_comp};
}

/**
 *
 * \author Kelton Halbert - NWS Storm Prediction Center
 *
 * \brief Conveniance function to compute the U and V components<!--
 * --> of a wind vector.
 *
 * Given the wind speed and direction of a vector via sharp::WindVector,
 * compute and return the zonal and meridional vector components as a struct.
 *
 * \param   vect    {speed, direction}
 *
 * \return  {u_comp, v_comp}
 */
[[nodiscard]] SHARP_HD SHARP_FORCEINLINE WindComponents vector_to_components(WindVector vect) {
#ifndef NO_QC
    if ((vect.speed == MISSING) || (vect.direction == MISSING))
        return {MISSING, MISSING};
#endif
    const float wind_direction = vect.direction * (PI / 180.0f);
    const float u_comp = -1.0f * vect.speed * std::sin(wind_direction);
    const float v_comp = -1.0f * vect.speed * std::cos(wind_direction);

    return {u_comp, v_comp};
}

/**
 *
 * \author Kelton Halbert - NWS Storm Prediction Center
 *
 * \brief Compute the mean wind over the given sharp::PressureLayer
 *
 * Computes the mean wind over the given sharp::PressureLayer
 * and arrays of pressure and wind components with a length of N.
 *
 *
 * \param   layer       {bottom, top}
 * \param   pres        (Pa)
 * \param   u_wind      (m/s)
 * \param   v_wind      (m/s)
 * \param   N           (length of arrays)
 * \param   weighted    (whether to weight by pressure level)
 *
 * \return  {mean_u, mean_v}
 */
template <typename PresPtr = const float*, typename UPtr = const float*, typename VPtr = const float*>
[[nodiscard]] SHARP_HD inline WindComponents mean_wind(PressureLayer layer, const PresPtr pressure,
                                                       const UPtr u_wind,
                                                       const VPtr v_wind,
                                                       const std::ptrdiff_t N,
                                                       const bool weighted) {
#ifndef NO_QC
    if ((layer.bottom == MISSING) || (layer.top == MISSING))
        return {MISSING, MISSING};
#endif

    if (layer.bottom > pressure[0]) layer.bottom = pressure[0];
    if (layer.top < pressure[N - 1]) layer.top = pressure[N - 1];

    LayerIndex idx = get_layer_index(layer, pressure, N);

    float u_lyr_bot = interp_pressure(layer.bottom, pressure, u_wind, N);
    float v_lyr_bot = interp_pressure(layer.bottom, pressure, v_wind, N);
    float u_lyr_top = interp_pressure(layer.top, pressure, u_wind, N);
    float v_lyr_top = interp_pressure(layer.top, pressure, v_wind, N);

#ifndef NO_QC
    if ((u_lyr_bot == MISSING) || (v_lyr_bot == MISSING) ||
        (u_lyr_top == MISSING) || (v_lyr_top == MISSING)) {
        return {MISSING, MISSING};
    }
#endif

    float u_integ = 0.0f;
    float v_integ = 0.0f;
    float w_integ = 0.0f;
    float dummy_w = 0.0f;

    float p_prev = layer.bottom;
    float u_prev = u_lyr_bot;
    float v_prev = v_lyr_bot;

    for (std::ptrdiff_t k = idx.kbot; k <= idx.ktop; ++k) {
#ifndef NO_QC
        if ((u_wind[k] == MISSING) || (v_wind[k] == MISSING)) {
            continue;
        }
#endif
        const float p_curr = pressure[k];
        const float u_curr = u_wind[k];
        const float v_curr = v_wind[k];

        if (weighted) {
            u_integ += _integ_trapz(u_curr * p_curr, u_prev * p_prev, p_curr,
                                    p_prev, dummy_w, false);
            v_integ += _integ_trapz(v_curr * p_curr, v_prev * p_prev, p_curr,
                                    p_prev, dummy_w, false);
            w_integ +=
                _integ_trapz(p_curr, p_prev, p_curr, p_prev, dummy_w, false);
        } else {
            u_integ +=
                _integ_trapz(u_curr, u_prev, p_curr, p_prev, w_integ, true);
            v_integ +=
                _integ_trapz(v_curr, v_prev, p_curr, p_prev, dummy_w, false);
        }

        p_prev = p_curr;
        u_prev = u_curr;
        v_prev = v_curr;
    }

    {
        if (weighted) {
            u_integ += _integ_trapz(u_lyr_top * layer.top, u_prev * p_prev,
                                    layer.top, p_prev, dummy_w, false);
            v_integ += _integ_trapz(v_lyr_top * layer.top, v_prev * p_prev,
                                    layer.top, p_prev, dummy_w, false);
            w_integ += _integ_trapz(layer.top, p_prev, layer.top, p_prev,
                                    dummy_w, false);
        } else {
            u_integ += _integ_trapz(u_lyr_top, u_prev, layer.top, p_prev,
                                    w_integ, true);
            v_integ += _integ_trapz(v_lyr_top, v_prev, layer.top, p_prev,
                                    dummy_w, false);
        }
    }

    return {u_integ / w_integ, v_integ / w_integ};
}

/**
 * \author Kelton Halbert - NWS Storm Prediction Center
 *
 * \brief Compute the max wind over a height or pressure layer
 *
 * Finds the maximum wind speed over a given sharp::PressureLayer or
 * sharp::HeightLayer, returning the vector components.
 *
 * \param   layer       {bottom, top}
 * \param   coordinate  (pressure (Pa) or height (m AGL))
 * \param   u_wind      (m/s)
 * \param   v_wind      (m/s)
 * \param   N           (length of arrays)
 *
 * \return {max_u, max_v}
 */
template <typename Layer, typename CoordPtr = const float*, typename UPtr = const float*, typename VPtr = const float*>
[[nodiscard]] SHARP_HD inline WindComponents max_wind(Layer layer, const CoordPtr coordinate,
                                                      const UPtr u_wind,
                                                      const VPtr v_wind,
                                                      const std::ptrdiff_t N) {
    LayerIndex lyr_idx = get_layer_index(layer, coordinate, N);

    float u_bot, u_top;
    float v_bot, v_top;
    if constexpr (layer.coord == LayerCoordinate::pressure) {
        if (layer.bottom > coordinate[0]) layer.bottom = coordinate[0];
        if (layer.top < coordinate[N - 1]) layer.top = coordinate[N - 1];

        u_bot = interp_pressure(layer.bottom, coordinate, u_wind, N);
        v_bot = interp_pressure(layer.bottom, coordinate, v_wind, N);
        u_top = interp_pressure(layer.top, coordinate, u_wind, N);
        v_top = interp_pressure(layer.top, coordinate, v_wind, N);
    } else {
        layer.bottom += coordinate[0];
        layer.top += coordinate[0];

        if (layer.bottom < coordinate[0]) layer.bottom = coordinate[0];
        if (layer.top > coordinate[N - 1]) layer.top = coordinate[N - 1];

        u_bot = interp_height(layer.bottom, coordinate, u_wind, N);
        v_bot = interp_height(layer.bottom, coordinate, v_wind, N);
        u_top = interp_height(layer.top, coordinate, u_wind, N);
        v_top = interp_height(layer.top, coordinate, v_wind, N);
    }

    float u_max = MISSING;
    float v_max = MISSING;
    float max = MISSING;
    if ((u_bot != MISSING) && (v_bot != MISSING)) {
        max = vector_magnitude(u_bot, v_bot);
        u_max = u_bot;
        v_max = v_bot;
    }

    for (std::ptrdiff_t idx = lyr_idx.kbot; idx <= lyr_idx.ktop; ++idx) {
        if ((u_wind[idx] == MISSING) || (v_wind[idx] == MISSING)) continue;
        float wspd = vector_magnitude(u_wind[idx], v_wind[idx]);
        if (wspd > max) {
            max = wspd;
            u_max = u_wind[idx];
            v_max = v_wind[idx];
        }
    }

    if ((u_top != MISSING) && (v_top != MISSING)) {
        float wspd = vector_magnitude(u_top, v_top);
        if (wspd > max) {
            max = wspd;
            u_max = u_top;
            v_max = v_top;
        }
    }

    return {u_max, v_max};
}

/**
 *
 * \author Kelton Halbert - NWS Storm Prediction Center
 *
 * \brief Compue the wind shear for a given layer
 *
 * Computes the U and V components of the wind shear over a
 * layer given the vertical sounding arrays
 * of pressure/height, u_wind, v_wind, and their length.
 *
 * This generic template handles the logic
 * at compike time on whether or not this is a sharp::HeightLayer
 * or a sharp::PressureLayer, and calls the interpolation routines
 * accordingly
 *
 * \param   layer   {bottom, top}
 * \param   coord   (meters or Pa)
 * \param   u_wind  (m/s)
 * \param   v_wind  (m/s)
 * \param   N       (length of arrays)
 *
 * \return  {shear_u, shear_v}
 */
template <typename L, typename CoordPtr = const float*, typename UPtr = const float*, typename VPtr = const float*>
[[nodiscard]] SHARP_HD constexpr inline WindComponents wind_shear(L layer, const CoordPtr coord,
                                                                  const UPtr u_wind,
                                                                  const VPtr v_wind,
                                                                  const std::ptrdiff_t N) {
#ifndef NO_QC
    if ((layer.bottom == MISSING) || (layer.top == MISSING))
        return {MISSING, MISSING};
#endif

    float u_bot = MISSING;
    float u_top = MISSING;
    float v_bot = MISSING;
    float v_top = MISSING;
    // This if statement disappears at compile time depending
    // on which kind of layer is passed
    if constexpr (layer.coord == LayerCoordinate::pressure) {
        if (layer.bottom > coord[0]) layer.bottom = coord[0];
        if (layer.top < coord[N - 1]) layer.top = coord[N - 1];

        u_bot = interp_pressure(layer.bottom, coord, u_wind, N);
        u_top = interp_pressure(layer.top, coord, u_wind, N);

        v_bot = interp_pressure(layer.bottom, coord, v_wind, N);
        v_top = interp_pressure(layer.top, coord, v_wind, N);
    } else {
        // AGL to MSL
        layer.bottom += coord[0];
        layer.top += coord[0];

        if (layer.bottom < coord[0]) layer.bottom = coord[0];
        if (layer.top > coord[N - 1]) layer.top = coord[N - 1];

        u_bot = interp_height(layer.bottom, coord, u_wind, N);
        u_top = interp_height(layer.top, coord, u_wind, N);

        v_bot = interp_height(layer.bottom, coord, v_wind, N);
        v_top = interp_height(layer.top, coord, v_wind, N);
    }

#ifndef NO_QC
    if ((u_bot == MISSING) || (v_bot == MISSING) || (u_top == MISSING) ||
        (v_top == MISSING)) {
        return {MISSING, MISSING};
    }
#endif

    return {u_top - u_bot, v_top - v_bot};
}

/**
 * \author Kelton Halbert - NWS Storm Prediction Center
 *
 * \brief Computes the Storm Relative Helicity (SRH) over a given layer
 *
 * Computes the Storm Relative Helicity (SRH) over a given layer
 * using storm motion vector components stored in sharp::WindComponents.
 * This tempalte is the generalized form that will fill in the appropriate
 * interpolation calls and coordinate arrays depending on whether a
 * sharp::PressureLayer or sharp::HeightLayer gets passed to this
 * template. The other API instances are just static instantiations
 * of this template.
 *
 * This integration occurs over the given arrays of coord, u_wind, and v_wind,
 * with N elements in each. The integration only uses interpolation
 * for the top and bottom of the specified layer.
 *
 * If using a sharp::HeightLayer, units are expected in above-ground-level
 * (AGL), and gets converted to above-mean-sea-level (MSL) during the
 * computation.
 *
 * \param   layer           {bottom, top}
 * \param   storm_motion    {storm_u, storm_v}
 * \param   coord           (meters or Pa)
 * \param   u_wind          (m/s)
 * \param   v_wind          (m/s)
 * \param   N               (length of arrays)
 *
 * \return  storm_relative_helicity
 */
template <typename L, typename CoordPtr = const float*, typename UPtr = const float*, typename VPtr = const float*>
[[nodiscard]] SHARP_HD inline float helicity(L layer, WindComponents storm_motion,
                                             const CoordPtr coord, const UPtr u_wind,
                                             const VPtr v_wind, const std::ptrdiff_t N) {
#ifndef NO_QC
    if ((storm_motion.u == MISSING) || (storm_motion.v == MISSING)) {
        return MISSING;
    }
    if ((layer.bottom == MISSING) || (layer.top == MISSING)) {
        return MISSING;
    }
#endif

    float sru_bot;
    float srv_bot;
    if constexpr (layer.coord == LayerCoordinate::height) {
        // get the height in MSL by adding the surface height
        layer.bottom += coord[0];
        layer.top += coord[0];

        // Get the interpolated bottom of the layer to integrate
        // and convert to storm-relative winds
        sru_bot = interp_height(layer.bottom, coord, u_wind, N);
        srv_bot = interp_height(layer.bottom, coord, v_wind, N);
    } else {
        // Get the interpolated bottom of the layer to integrate
        // and convert to storm-relative winds
        sru_bot = interp_pressure(layer.bottom, coord, u_wind, N);
        srv_bot = interp_pressure(layer.bottom, coord, v_wind, N);
    }
    sru_bot -= storm_motion.u;
    srv_bot -= storm_motion.v;

    // Get the vertical array indices corresponding to our layer,
    // while also bounds checking our search. Indices exclude the
    // top and bottom layers that will be interpolated.
    LayerIndex layer_idx = get_layer_index(layer, coord, N);

    // will get set in first loop iter
    float sru_top;
    float srv_top;
    float layer_helicity = 0.0f;
    for (std::ptrdiff_t k = layer_idx.kbot; k <= layer_idx.ktop; ++k) {
#ifndef NO_QC
        if ((u_wind[k] == MISSING) || (v_wind[k] == MISSING)) {
            continue;
        }
#endif
        // top of layer storm relative winds
        sru_top = u_wind[k] - storm_motion.u;
        srv_top = v_wind[k] - storm_motion.v;

        // integrate layer
        layer_helicity += (sru_top * srv_bot) - (sru_bot * srv_top);
        // set the top to be the bottom
        // of the next layer
        sru_bot = sru_top;
        srv_bot = srv_top;
    }

    // Get the interpolated top of the layer to integrate
    // and convert to storm-relative winds
    if constexpr (layer.coord == LayerCoordinate::height) {
        sru_top = interp_height(layer.top, coord, u_wind, N);
        srv_top = interp_height(layer.top, coord, v_wind, N);
    } else {
        sru_top = interp_pressure(layer.top, coord, u_wind, N);
        srv_top = interp_pressure(layer.top, coord, v_wind, N);
    }
    sru_top -= storm_motion.u;
    srv_top -= storm_motion.v;

    // integrate the final layer
    layer_helicity += (sru_top * srv_bot) - (sru_bot * srv_top);

    return layer_helicity;
}

}  // end namespace sharp

#endif  // SHARP_WINDS_H
