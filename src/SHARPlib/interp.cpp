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

#include <SHARPlib/interp.h>

namespace sharp {

float interp_height(const float height_val, const float height_arr[],
                    const float data_arr[], const std::ptrdiff_t N) {
    return interp_height<const float*, const float*>(height_val, height_arr, data_arr, N);
}

float interp_pressure(const float pressure_val, const float pressure_arr[],
                      const float data_arr[], const std::ptrdiff_t N) {
    return interp_pressure<const float*, const float*>(pressure_val, pressure_arr, data_arr, N);
}

float find_first_pressure(const float data_val, const float pressure_arr[],
                          const float data_arr[], const std::ptrdiff_t N) {
    return find_first_pressure<const float*, const float*>(data_val, pressure_arr, data_arr, N);
}

float find_first_height(const float data_val, const float height_arr[],
                        const float data_arr[], const std::ptrdiff_t N) {
    return find_first_height<const float*, const float*>(data_val, height_arr, data_arr, N);
}

}  // end namespace sharp
