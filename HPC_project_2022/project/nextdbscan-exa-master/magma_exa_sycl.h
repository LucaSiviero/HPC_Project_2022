/**
Copyright (c) 2021, Ernir Erlingsson
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
 */

#ifndef NEXTDBSCAN20_MAGMA_EXA_CU_H
#define NEXTDBSCAN20_MAGMA_EXA_CU_H
/*
Luca Siviero: Defining _ONEDPL_BACKEND_SYCL because it's written in the file that gives me error.
*/
#ifndef _ONEDPL_BACKEND_SYCL
#define _ONEDPL_BACKEND_SYCL 1
#endif

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cassert>
#include <iostream>
#include <dpct/dpl_utils.hpp>
#include <numeric>

template <typename T> using h_vec = std::vector<T>;
template <typename T1> using d_vector = dpct::device_vector<T1>;
namespace exa {

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void fill(d_vector<T> &v, std::size_t const begin, std::size_t const end, T const val) noexcept {
#ifdef DEBUG_ON
        assert(begin <= end);
#endif
        std::fill(oneapi::dpl::execution::make_device_policy(
                      dpct::get_default_queue()),
                  v.begin() + begin, v.begin() + end, val);
    }

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void iota(d_vector<T> &v, std::size_t const begin, std::size_t const end, std::size_t const startval) noexcept {
#ifdef DEBUG_ON
        assert(begin <= end);
#endif
        dpct::iota(oneapi::dpl::execution::make_device_policy(
                       dpct::get_default_queue()),
                   v.begin() + begin, v.begin() + end, startval);
    }

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    T reduce(d_vector<T> &v, std::size_t const begin, std::size_t const end, T const startval) noexcept {
#ifdef DEBUG_ON
        assert(begin <= end);
        assert((end - begin) <= (v.size() - begin));
#endif
        return std::reduce(oneapi::dpl::execution::make_device_policy(
                               dpct::get_default_queue()),
                           v.begin() + begin, v.begin() + end, startval);
    }

        /*fLuca Siviero: I have personally modiied this function. I don't know if it actually works.
        The thrust::advance(iter, n) function increases the iterator n times if n is positive, otherwise it decreases it.
        */
    template <typename T, typename F, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
        std::size_t count_if(d_vector<T> &v, std::size_t const begin, std::size_t const end, F const &functor) noexcept {
#ifdef DEBUG_ON
        assert(begin <= end);
#endif
        auto iter1 = 1;//v.begin();
        auto iter2 = 1;//v.begin();
        //std::cout << "begin: " << *v.begin() << " end: " << end << std::endl;
        //std::advance(iter1, begin);
        //std::advance(iter2, begin);
        iter1 += begin;
        iter2 += end;
        //std::cout << "After std::advance iter1 is: " << *iter1 << ", begin is: "<< begin << ", iter2 is: " << *iter2 << std::endl;
        //thrust::advance(iter2, end);
//        return thrust::count_if(v.begin() + begin, v.begin() + end, functor);
        
        return std::count_if(oneapi::dpl::execution::make_device_policy(
                                 dpct::get_default_queue()),
                             v.begin() + begin, v.begin() + end, functor);
        
       //return std::count_if(v.begin() + begin, v.begin() + end, functor);
    }

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void exclusive_scan(d_vector<T> const &v_input, std::size_t const in_begin, std::size_t const in_end,          //Changing dpct::device iterator to const device_iterator<int>
            d_vector<T> &v_output, std::size_t const out_begin, T const init) noexcept {
#ifdef DEBUG_ON
        assert(in_begin <= in_end);
#endif
        /* Luca Siviero: Changing the functions to usa only std::exclusive_scan without dpct queue. Probably offsets are needed.
        std::exclusive_scan(oneapi::dpl::execution::make_device_policy(
                                dpct::get_default_queue()),
                            v_input.begin() + in_begin,
                            v_input.begin() + in_end,
                            v_output.begin() + out_begin, init, 0);
        */
       //auto in_begin2 = std::distance(v_input.begin(), in_begin);
       /* Luca: Changin exclusive_scan to use only the device_vector. */
       std::exclusive_scan( oneapi::dpl::execution::make_device_policy(
                            dpct::get_default_queue()),
                            v_input.begin() + in_begin,     //in_begin, in_ed, out_begin added... Yet output is the same as before
                            v_input.end() + in_begin,
                            v_output.begin() + out_begin, 0);
    }

        /*
        Luca Siviero: The function has been modified by me because the v_output.resize() was using a thrust argument. I changed it to use std::distance as argument for the resize functions, instead of thrust::distance.
        */
    template <typename T, typename F, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void copy_if(d_vector<T> &v_input, std::size_t const in_begin, std::size_t const in_end, d_vector<T> &v_output,
            std::size_t const out_begin, F const &functor) noexcept {
#ifdef DEBUG_ON
        assert(in_begin <= in_end);
#endif
        
        auto it =
            std::copy_if(oneapi::dpl::execution::make_device_policy(
                             dpct::get_default_queue()),
                         v_input.begin() + in_begin, v_input.begin() + in_end,
                         v_output.begin() + out_begin, functor);
        
        //std::cout << "Call to copy if finished and returned " << *it << std::endl;
    
        v_output.resize(std::distance(v_output.begin(), it));
    }

    template <typename F>
    void for_each(std::size_t const begin, std::size_t const end, F const &functor) noexcept {
#ifdef DEBUG_ON
        assert(begin <= end);
#endif
//        thrust::counting_iterator<std::size_t> it_cnt_begin(begin);
//        thrust::counting_iterator<std::size_t> it_cnt_end = it_cnt_begin + (end - begin);
//        thrust::for_each(it_cnt_begin, it_cnt_begin + (end - begin), functor);
//        thrust::make_counting_iterator(0),
        std::for_each(oneapi::dpl::execution::make_device_policy(
                          dpct::get_default_queue()),
                      dpct::make_counting_iterator(begin),
                      dpct::make_counting_iterator(begin + (end - begin)),
                      functor);
//        thrust::for_each(it_cnt_begin, it_cnt_end, functor);
    }

    template <typename F>
    void for_each_experimental(std::size_t const begin, std::size_t const end,
                               F const &functor) noexcept {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
#ifdef DEBUG_ON
        assert(begin <= end);
#endif
        oneapi::dpl::counting_iterator<int> it_cnt_begin(begin);
        oneapi::dpl::counting_iterator<int> it_cnt_end = it_cnt_begin + (end - begin);
        std::for_each(oneapi::dpl::execution::make_device_policy(q_ct1),
                      oneapi::dpl::execution::make_device_policy(q_ct1),
                      it_cnt_begin, it_cnt_end, [=](auto const &i) {
            functor(i);
                      });
    }

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void lower_bound(d_vector<T> &v_input, std::size_t const in_begin, std::size_t const in_end,
            d_vector<T> &v_value, std::size_t const value_begin, std::size_t const value_end,
            d_vector<int> &v_output, std::size_t const out_begin, int const stride) noexcept {
#ifdef DEBUG_ON
        assert(in_begin <= in_end);
        assert(value_begin <= value_end);
        assert(v_input.size() >= (in_end - in_begin));
        assert(v_output.size() >= ((value_end - value_begin - 1) * (stride + 1)) + out_begin);
#endif
//        thrust::counting_iterator<int> it_cnt_begin(in_begin);
        oneapi::dpl::counting_iterator<int> it_cnt_begin(0);
        auto it_trans_begin =
            oneapi::dpl::make_transform_iterator(it_cnt_begin, [=](auto _1) {
                                                                                              return (_1 * (stride + 1)) + out_begin;
            });
        auto it_perm_begin = oneapi::dpl::make_permutation_iterator(
            v_output.begin(), it_trans_begin);
        oneapi::dpl::lower_bound(oneapi::dpl::execution::make_device_policy(
                                     dpct::get_default_queue()),
                                 v_input.begin() + in_begin,
                                 v_input.begin() + in_end,
                                 v_value.begin() + value_begin,
                                 v_value.begin() + value_end, it_perm_begin);
    }

    template <typename T, typename F, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void lower_bound(d_vector<T> &v_input, std::size_t const in_begin, std::size_t const in_end,
            d_vector<T> &v_value, std::size_t const value_begin, std::size_t const value_end,
            d_vector<int> &v_output, std::size_t const out_begin, int const stride, F const &functor) noexcept {
#ifdef DEBUG_ON
        assert(in_begin <= in_end);
        assert(value_begin <= value_end);
        assert(v_input.size() >= (in_end - in_begin));
        assert(v_output.size() >= ((value_end - value_begin - 1) * (stride + 1)) + out_begin);
#endif
//        thrust::counting_iterator<int> it_cnt_begin(in_begin);
        oneapi::dpl::counting_iterator<int> it_cnt_begin(0);
        auto it_trans_begin =
            oneapi::dpl::make_transform_iterator(it_cnt_begin, [=](auto _1) {
                                                                                              return (_1 * (stride + 1)) + out_begin;
            });
        auto it_perm_begin = oneapi::dpl::make_permutation_iterator(
            v_output.begin(), it_trans_begin);
        oneapi::dpl::lower_bound(
            oneapi::dpl::execution::make_device_policy(
                dpct::get_default_queue()),
            v_input.begin() + in_begin, v_input.begin() + in_end,
            v_value.begin() + value_begin, v_value.begin() + value_end,
            it_perm_begin, functor);
    }



    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void upper_bound(d_vector<T> &v_input, std::size_t const in_begin, std::size_t const in_end,
            d_vector<T> &v_value, std::size_t const value_begin, std::size_t const value_end,
            d_vector<int> &v_output, std::size_t const out_begin, int const stride) noexcept {
#ifdef DEBUG_ON
        assert(in_begin <= in_end);
        assert(value_begin <= value_end);
        assert(v_input.size() >= (in_end - in_begin));
        assert(v_output.size() >= ((value_end - value_begin - 1) * (stride + 1)) + out_begin);
#endif
        oneapi::dpl::counting_iterator<int> it_cnt_begin(0);
        auto it_trans_begin =
            oneapi::dpl::make_transform_iterator(it_cnt_begin, [=](auto _1) {
                                                                                              return (_1 * (stride + 1)) + out_begin;
            });
        auto it_perm_begin = oneapi::dpl::make_permutation_iterator(
            v_output.begin(), it_trans_begin);
        oneapi::dpl::upper_bound(oneapi::dpl::execution::make_device_policy(
                                     dpct::get_default_queue()),
                                 v_input.begin() + in_begin,
                                 v_input.begin() + in_end,
                                 v_value.begin() + value_begin,
                                 v_value.begin() + value_end, it_perm_begin);
    }

    template <typename T, typename F, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void upper_bound(d_vector<T> &v_input, std::size_t const in_begin, std::size_t const in_end,
            d_vector<T> &v_value, std::size_t const value_begin, std::size_t const value_end,
            d_vector<int> &v_output, std::size_t const out_begin, int const stride, F const &functor) noexcept {
#ifdef DEBUG_ON
        assert(in_begin <= in_end);
        assert(value_begin <= value_end);
        assert(v_input.size() >= (in_end - in_begin));
        assert(v_output.size() >= ((value_end - value_begin - 1) * (stride + 1)) + out_begin);
#endif
        oneapi::dpl::counting_iterator<int> it_cnt_begin(0);
        auto it_trans_begin =
            oneapi::dpl::make_transform_iterator(it_cnt_begin, [=](auto _1) {
                                                                                              return (_1 * (stride + 1)) + out_begin;
            });
        auto it_perm_begin = oneapi::dpl::make_permutation_iterator(
            v_output.begin(), it_trans_begin);
        oneapi::dpl::upper_bound(
            oneapi::dpl::execution::make_device_policy(
                dpct::get_default_queue()),
            v_input.begin() + in_begin, v_input.begin() + in_end,
            v_value.begin() + value_begin, v_value.begin() + value_end,
            it_perm_begin, functor);
    }

    /*Luca Siviero: After some trouble trying to translate the minmax_function, I found out that the function was not used. I'm going to comment it.*/
/*
    template <typename T, typename F,
        typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr>
    std::pair<T, T> minmax_element(d_vector<T> &v, std::size_t const begin,
                                   std::size_t const end,
                                   F const &functor) noexcept {
#ifdef DEBUG_ON
        assert(begin <= end);
#endif
        auto pair = thrust::minmax_element(v.begin() + begin, v.begin() + end, functor);
        auto min = thrust::min_element(v.begin() + begin, v.begin() + end, functor);
        auto max = thrust::max_element(v.begin() + begin, v.begin() + end, functor);
        return std::pair(*min, *max);
    }
*/
    template <typename T, typename F, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void sort(d_vector<T> &v, std::size_t const begin, std::size_t const end, F const &functor) noexcept {
#ifdef DEBUG_ON
        assert(begin <= end);
        assert((end - begin) <= (v.size() - begin));
#endif
        oneapi::dpl::sort(oneapi::dpl::execution::make_device_policy(
                              dpct::get_default_queue()),
                          v.begin() + begin, v.begin() + end, functor);
    }

    template <typename T1, typename T2, typename F, typename std::enable_if<std::is_arithmetic<T1>::value>::type* = nullptr>
    void unique(d_vector<T1> &v_input, d_vector<T2> &v_output, std::size_t const in_begin, std::size_t const in_end,
            std::size_t const out_begin, F const &functor) noexcept {
#ifdef DEBUG_ON
        assert(in_begin <= in_end);
#endif
    }

    template <typename T1, typename T2, typename F, typename std::enable_if<std::is_arithmetic<T1>::value>::type* = nullptr>
    void transform(d_vector<T1> const &v_input, std::size_t const in_begin, std::size_t const in_end, d_vector<T2> &v_output,
            std::size_t const out_begin, F const &functor) noexcept {
#ifdef DEBUG_ON
        assert(in_begin <= in_end);
#endif
        std::transform(oneapi::dpl::execution::make_device_policy(
                           dpct::get_default_queue()),
                       v_input.begin() + in_begin, v_input.begin() + in_end,
                       v_output.begin() + out_begin, functor);
    }

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr>

    void atomic_add(dpct::device_pointer<T> v, T const val) {
        T *ptr = dpct::get_raw_pointer(v);
        atomicAdd(ptr, val);
    }
    
    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr>

    /*
    Luca: We changed this function because it was still using atomicMin from CUDA.
    */
    void atomic_min(/*dpct::device_pointer<T> v*/ T* v, T val) {
        T *ptr = dpct::get_raw_pointer(v);
        //atomicMin(ptr, val);
        std::min(*ptr, val);
    }

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void copy(d_vector<T> const &v_input, std::size_t const in_begin, std::size_t const in_end, d_vector<T> &v_output,
            std::size_t const out_begin) {
                std::cout << "EXECUTING COPY ON DEVICE:";
                std::cout << dpct::get_default_queue().get_device().get_info<cl::sycl::info::device::name>() << std::endl;
                std::cout << "WITH THIS MAXIMUM COMPUTE UNITS" << std::endl;
                std::cout << dpct::get_default_queue().get_device().get_info<cl::sycl::info::device::max_compute_units>() << std::endl;
        std::copy(oneapi::dpl::execution::make_device_policy(
                      dpct::get_default_queue()),
                  v_input.begin() + in_begin, v_input.begin() + in_end,
                  v_output.begin() + out_begin);
    }

};
#endif //NEXTDBSCAN20_MAGMA_EXA_CU_H
