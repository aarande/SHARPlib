/**
 * \file
 * \brief Utilities and structures useful for grouping objects,<!--
 * --> conducting QC, and things that otherwise don't have a home.
 * \author
 *   Kelton Halbert                  \n
 *   Email: kelton.halbert@noaa.gov  \n
 * \date   2022-11-02
 *
 * Written for the NWS Storm Predidiction Center \n
 * Based on NSHARP routines originally written by
 * John Hart and Rich Thompson at SPC.
 */
#include <SHARPlib/layer.h>

namespace sharp {

LayerIndex get_layer_index(PressureLayer& layer, const float pressure[],
                           const std::ptrdiff_t N) {
    return get_layer_index<const float*>(layer, pressure, N);
}

LayerIndex get_layer_index(HeightLayer& layer, const float height[],
                           const std::ptrdiff_t N) {
    return get_layer_index<const float*>(layer, height, N);
}

PressureLayer height_layer_to_pressure(HeightLayer layer,
                                       const float pressure[],
                                       const float height[],
                                       const std::ptrdiff_t N,
                                       const bool isAGL) {
    return height_layer_to_pressure<const float*, const float*>(layer, pressure, height, N, isAGL);
}

HeightLayer pressure_layer_to_height(PressureLayer layer,
                                     const float pressure[],
                                     const float height[],
                                     const std::ptrdiff_t N, const bool toAGL) {
    return pressure_layer_to_height<const float*, const float*>(layer, pressure, height, N, toAGL);
}

float layer_mean(PressureLayer layer, const float pressure[],
                 const float data_arr[], const std::ptrdiff_t N) {
    return layer_mean<const float*, const float*>(layer, pressure, data_arr, N);
}

float layer_mean(HeightLayer layer, const float height[],
                 const float pressure[], const float data_arr[],
                 const std::ptrdiff_t N, const bool isAGL) {
    return layer_mean<const float*, const float*, const float*>(layer, height, pressure, data_arr, N, isAGL);
}

/// @cond DOXYGEN_IGNORE

template float layer_min<PressureLayer>(PressureLayer layer,
                                        const float coord_arr[],
                                        const float data_arr[],
                                        const std::ptrdiff_t N,
                                        float* lvl_of_min);

template float layer_min<HeightLayer>(HeightLayer layer,
                                      const float coord_arr[],
                                      const float data_arr[],
                                      const std::ptrdiff_t N,
                                      float* lvl_of_min);

template float layer_max<PressureLayer>(PressureLayer layer,
                                        const float coord_arr[],
                                        const float data_arr[],
                                        const std::ptrdiff_t N,
                                        float* lvl_of_max);

template float layer_max<HeightLayer>(HeightLayer layer,
                                      const float coord_arr[],
                                      const float data_arr[],
                                      const std::ptrdiff_t N,
                                      float* lvl_of_max);

template float integrate_layer_trapz<PressureLayer>(
    PressureLayer layer, const float var_array[], const float coord_array[],
    const std::ptrdiff_t N, const int integ_sign, const bool weighted);

template float integrate_layer_trapz<HeightLayer>(
    HeightLayer layer, const float var_array[], const float coord_array[],
    const std::ptrdiff_t N, const int integ_sign, const bool weighted);

/// @endcond

}  // end namespace sharp
