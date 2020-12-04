//
// Created by zc on 20-11-21.
//
#pragma once
#ifndef MOBILEFACENET_AS_FACEMASK_H
#define MOBILEFACENET_AS_FACEMASK_H
#include "string"
#include "net.h"
#include "iostream"
#include "opencv.h"

namespace Face{
    class facemask {
    public:
        facemask(const std::string &model_path);
        ~facemask();
        void start(ncnn::Mat& ncnn_img, std::vector<float>&feature56);
        double maskNet(ncnn::Mat& img_);
        ncnn::Net masknet;

    private:
        int threadnum = 1;
    };
    }



#endif //MOBILEFACENET_AS_FACEMASK_H
