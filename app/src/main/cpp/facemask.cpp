//
// Created by zc on 20-11-21.
//
#include "facemask.h"

namespace Face {
    facemask::facemask(const std::string &model_path) {
        std::string param_files = model_path + "/mask.param";
        std::string bin_files = model_path + "/mask.bin";
        masknet.load_param(param_files.c_str());
        masknet.load_model(bin_files.c_str());
    }
    facemask::~facemask() {
        masknet.clear();
    }

   double facemask::maskNet(ncnn::Mat& img_) {
        const float mean_vals[3] = { 127.5f, 127.5f, 127.5f };
        const float e_vals[3] = { 1.0f/128.0f,1.0f/128.0f,1.0f/128.0f };
        img_.substract_mean_normalize(mean_vals, e_vals);
        ncnn::Extractor ex = masknet.create_extractor();
        ex.set_num_threads(threadnum);
        ex.set_light_mode(true);
        ex.input("input.1", img_);
        ncnn::Mat out;
        ex.extract("522", out);
//        std::string labels[2]={"mask","no_mask"};
        double outs=0.0;
        if (out[0]>out[1]){
            outs=0;
        }
        else {
            outs=1;
        }
        return outs;
    }
}