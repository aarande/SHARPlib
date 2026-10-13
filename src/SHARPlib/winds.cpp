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
#include <SHARPlib/constants.h>
#include <SHARPlib/interp.h>
#include <SHARPlib/layer.h>
#include <SHARPlib/winds.h>

#include <cmath>
#include <cstddef>

#include "SHARPlib/algorithms.h"

namespace sharp {

WindComponents mean_wind(PressureLayer layer, const float pressure[],
                         const float u_wind[], const float v_wind[],
                         const std::ptrdiff_t N, const bool weighted) {
    return mean_wind<const float*, const float*, const float*>(
        layer, pressure, u_wind, v_wind, N, weighted);
}

/// @cond DOXYGEN_IGNORE

template WindComponents mean_wind<const float*, const float*, const float*>(
    PressureLayer layer, const float pressure[], const float u_wind[],
    const float v_wind[], const std::ptrdiff_t N, const bool weighted);

template WindComponents max_wind<PressureLayer, const float*, const float*, const float*>(
    PressureLayer lyr, const float coordinate[], const float u_wind[],
    const float v_wind[], const std::ptrdiff_t N);

template WindComponents max_wind<HeightLayer, const float*, const float*, const float*>(
    HeightLayer lyr, const float coordinate[], const float u_wind[],
    const float v_wind[], const std::ptrdiff_t N);

template WindComponents wind_shear<PressureLayer, const float*, const float*, const float*>(
    PressureLayer layer, const float coord[], const float u_wind[],
    const float v_wind[], const std::ptrdiff_t N);

template WindComponents wind_shear<HeightLayer, const float*, const float*, const float*>(
    HeightLayer layer, const float coord[], const float u_wind[],
    const float v_wind[], const std::ptrdiff_t N);

template float helicity<PressureLayer, const float*, const float*, const float*>(
    PressureLayer layer, WindComponents storm_motion, const float coord[],
    const float u_wind[], const float v_wind[], const std::ptrdiff_t N);

template float helicity<HeightLayer, const float*, const float*, const float*>(
    HeightLayer layer, WindComponents storm_motion, const float coord[],
    const float u_wind[], const float v_wind[], const std::ptrdiff_t N);

/// @endcond

}  // end namespace sharp
