#include <android/bitmap.h>
#include <android/log.h>
#include <jni.h>
#include <string>
#include <vector>
#include <cstring>

// ncnn
#include "net.h"
#include "recognize.h"
#include "detect.h"
#include "retinaface.h"
#include "facemask.h"
using namespace Face;

#define TAG "MtcnnSo"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG,TAG,__VA_ARGS__)
//mtcnn static class
static Detect *mDetect;
static Recognize *mRecognize;
static facemask *mMaskface;
//retina static class

//sdk是否初始化成功
bool detection_sdk_init_ok = false;
static RetinaFaceNet *retinafacenet;
static std::vector<unsigned char> faceDetectParams;
static std::vector<unsigned char> faceDetectBin;

extern "C" {


JNIEXPORT jboolean JNICALL
Java_com_aiface_1as_Face_RemnnFaceDetectionModelInit(JNIEnv *env, jobject instance,
                                                           jstring faceDetectionModelPath_) {
    LOGD("JNI开始人脸检测模型初始化");
    //如果已初始化则直接返回
    if (detection_sdk_init_ok) {
        LOGD("人脸检测模型已经导入");
        return true;
    }
    jboolean tRet = false;
    if (NULL == faceDetectionModelPath_) {
        LOGD("导入的人脸检测的目录为空");
        return tRet;
    }

    //获取MTCNN模型的绝对路径的目录（不是/aaa/bbb.bin这样的路径，是/aaa/)
    const char *faceDetectionModelPath = env->GetStringUTFChars(faceDetectionModelPath_, 0);
    if (NULL == faceDetectionModelPath) {
        return tRet;
    }

    string tFaceModelDir = faceDetectionModelPath;
    string tLastChar = tFaceModelDir.substr(tFaceModelDir.length() - 1, 1);
    //LOGD("init, tFaceModelDir last =%s", tLastChar.c_str());
    //目录补齐/
    if ("\\" == tLastChar) {
        tFaceModelDir = tFaceModelDir.substr(0, tFaceModelDir.length() - 1) + "/";
    } else if (tLastChar != "/") {
        tFaceModelDir += "/";
    }
    LOGD("init, tFaceModelDir=%s", tFaceModelDir.c_str());

    //没判断是否正确导入
    LOGD("导入的人retinnaface---------------succession-----------------------");
//    mDetect = new Detect(tFaceModelDir);
    retinafacenet = new RetinaFaceNet(tFaceModelDir);  //
    LOGD("导入的人retinnaface---------------succession-----------------------");
    mRecognize = new Recognize(tFaceModelDir);
    mMaskface= new facemask(tFaceModelDir);    //添加 分类模型init
//    mDetect->SetMinFace(40);
//    mDetect->SetNumThreads(2);    // 2线程
    mRecognize->SetThreadNum(2);

    env->ReleaseStringUTFChars(faceDetectionModelPath_, faceDetectionModelPath);
    detection_sdk_init_ok = true;
//    LOGD("人脸检测模型已经导入");
    tRet = true;
    return tRet;
}

JNIEXPORT jintArray JNICALL
Java_com_aiface_1as_Face_ReFaceDetection(JNIEnv *env, jobject instance, jbyteArray imageDate_,
                                               jint imageWidth, jint imageHeight, jint imageChannel) {
    //  LOGD("JNI开始检测人脸");
    if(!detection_sdk_init_ok){
        LOGD("人脸检测RetinaFace模型SDK未初始化，直接返回空");
        return NULL;
    }

    int tImageDateLen = env->GetArrayLength(imageDate_);
    if(imageChannel == tImageDateLen / imageWidth / imageHeight){
        LOGD("数据宽=%d,高=%d,通道=%d",imageWidth,imageHeight,imageChannel);
    }
    else{
        LOGD("数据长宽高通道不匹配，直接返回空");
        return NULL;
    }

    jbyte *imageDate = env->GetByteArrayElements(imageDate_, NULL);
    if (NULL == imageDate){
        LOGD("导入数据为空，直接返回空");
        env->ReleaseByteArrayElements(imageDate_, imageDate, 0);
        return NULL;
    }

    if(imageWidth<20||imageHeight<20){
        LOGD("导入数据的宽和高小于20，直接返回空");
        env->ReleaseByteArrayElements(imageDate_, imageDate, 0);
        return NULL;
    }

    //TODO 通道需测试
    if(3 == imageChannel || 4 == imageChannel){
        //图像通道数只能是3或4；
    }else{
        LOGD("图像通道数只能是3或4，直接返回空");
        env->ReleaseByteArrayElements(imageDate_, imageDate, 0);
        return NULL;
    }

    //int32_t minFaceSize=40;
    //RetinaFace->SetMinFace(minFaceSize);

    unsigned char *faceImageCharDate = (unsigned char*)imageDate;
    ncnn::Mat ncnn_img;
    if(imageChannel==3) {
        ncnn_img = ncnn::Mat::from_pixels(faceImageCharDate, ncnn::Mat::PIXEL_BGR2RGB,
                                          imageWidth, imageHeight);
    }else{
        //ncnn_img = ncnn::Mat::from_pixels(faceImageCharDate, ncnn::Mat::PIXEL_RGBA2RGB, imageWidth, imageHeight);
        ncnn_img = ncnn::Mat::from_pixels(faceImageCharDate, ncnn::Mat::PIXEL_RGBA2RGB, imageWidth, imageHeight);
    }

    //std::vector<Bbox> finalBbox;
    std::vector<FaceObject> faceobjects;
    LOGD("检测到的人脸数目----：%d\n", ncnn_img.dims);
    retinafacenet->detect(ncnn_img, faceobjects);

    //int32_t num_face = static_cast<int32_t>(finalBbox.size());
    int32_t num_face = static_cast<int32_t>(faceobjects.size());
    LOGD("检测到的人脸数目----：%d\n", num_face);

    int out_size = 1+num_face*14;

    //  LOGD("内部人脸检测完成,开始导出数据");
    int *faceInfo = new int[out_size];

    faceInfo[0] = num_face;
    LOGD("检测到的人脸数目faceInfo[0] = %d\n", faceInfo[0]);
    LOGD("检测到的人脸数目faceInfo.size = %d\n", sizeof(faceInfo)/sizeof(&faceInfo[0]) );

    LOGD("landmark[0].x = %d\n", static_cast<int>(faceobjects[0].landmark[0].x));
    LOGD("landmark[0].y = %d\n", static_cast<int>(faceobjects[0].landmark[0].y));

    LOGD("landmark[4].x = %f\n", faceobjects[0].landmark[4].x);
    LOGD("landmark[4].y = %f\n", faceobjects[0].landmark[4].y);

    //faceobjects[0].landmark[0].x ;

    //(*faceobjects[0])->


    for(int i=0;i<num_face;i++){

        LOGD("---保存人脸数据---\n");

        float p = faceobjects[i].prob;
        LOGD(" ===faceobjects[i].prob = %f\n", p);

        faceInfo[14*i+1] = faceobjects[i].rect.x;//left
        faceInfo[14*i+2] = faceobjects[i].rect.y;//top
        faceInfo[14*i+3] = faceobjects[i].rect.x + faceobjects[i].rect.width;//right
        faceInfo[14*i+4] = faceobjects[i].rect.y + faceobjects[i].rect.height;//bottom

        LOGD("===faceobjects[i].rect.x = %f\n", faceobjects[i].rect.x);
        LOGD("===faceobjects[i].rect.y = %f\n", faceobjects[i].rect.y);

        faceInfo[14*i+5]  = static_cast<int>(faceobjects[i].landmark[0].x);
        faceInfo[14*i+6]  = static_cast<int>(faceobjects[i].landmark[1].x);
        faceInfo[14*i+7]  = static_cast<int>(faceobjects[i].landmark[2].x);
        faceInfo[14*i+8]  = static_cast<int>(faceobjects[i].landmark[3].x);
        faceInfo[14*i+9]  = static_cast<int>(faceobjects[i].landmark[4].x);

        faceInfo[14*i+10] = static_cast<int>(faceobjects[i].landmark[0].y);
        faceInfo[14*i+11] = static_cast<int>(faceobjects[i].landmark[1].y);
        faceInfo[14*i+12] = static_cast<int>(faceobjects[i].landmark[2].y);
        faceInfo[14*i+13] = static_cast<int>(faceobjects[i].landmark[3].y);
        faceInfo[14*i+14] = static_cast<int>(faceobjects[i].landmark[4].y);

//		for (int j =0;j<5;j++){
//            faceInfo[14*i+5 + 2*j]    =static_cast<int>(faceobjects[i].landmark[j].x);
//			faceInfo[14*i+5 + 2*j + 1]=static_cast<int>(faceobjects[i].landmark[j].y);
//        }
    }

    jintArray tFaceInfo = env->NewIntArray(out_size);
    env->SetIntArrayRegion(tFaceInfo,0,out_size,faceInfo);
    LOGD("内部人脸检测完成,导出数据成功");
    delete[] faceInfo;
    env->ReleaseByteArrayElements(imageDate_, imageDate, 0);
    return tFaceInfo;
}



JNIEXPORT jboolean JNICALL
Java_com_aiface_1as_Face_FaceDetectionModelInit(JNIEnv *env, jobject instance,
                                               jstring faceDetectionModelPath_) {
    LOGD("JNI开始人脸检测模型初始化");
    //如果已初始化则直接返回
    if (detection_sdk_init_ok) {
        LOGD("人脸检测模型已经导入");
        return true;
    }
    jboolean tRet = false;
    if (NULL == faceDetectionModelPath_) {
        LOGD("导入的人脸检测的目录为空");
        return tRet;
    }

    //获取MTCNN模型的绝对路径的目录（不是/aaa/bbb.bin这样的路径，是/aaa/)
    const char *faceDetectionModelPath = env->GetStringUTFChars(faceDetectionModelPath_, 0);
    if (NULL == faceDetectionModelPath) {
        return tRet;
    }

    string tFaceModelDir = faceDetectionModelPath;
    string tLastChar = tFaceModelDir.substr(tFaceModelDir.length() - 1, 1);
    //LOGD("init, tFaceModelDir last =%s", tLastChar.c_str());
    //目录补齐/
    if ("\\" == tLastChar) {
        tFaceModelDir = tFaceModelDir.substr(0, tFaceModelDir.length() - 1) + "/";
    } else if (tLastChar != "/") {
        tFaceModelDir += "/";
    }
    LOGD("init, tFaceModelDir=%s", tFaceModelDir.c_str());

    //没判断是否正确导入
    mDetect = new Detect(tFaceModelDir);
    retinafacenet = new RetinaFaceNet(tFaceModelDir);
    mRecognize = new Recognize(tFaceModelDir);
    mDetect->SetMinFace(40);
    mDetect->SetNumThreads(2);    // 2线程
    mRecognize->SetThreadNum(2);

    env->ReleaseStringUTFChars(faceDetectionModelPath_, faceDetectionModelPath);
    detection_sdk_init_ok = true;
//    LOGD("人脸检测模型已经导入");
    tRet = true;
    return tRet;
}

JNIEXPORT jintArray JNICALL
Java_com_aiface_1as_Face_FaceDetect(JNIEnv *env, jobject instance, jbyteArray imageDate_,
                                   jint imageWidth, jint imageHeight, jint imageChannel) {
    //  LOGD("JNI开始检测人脸");
    if(!detection_sdk_init_ok){
        LOGD("人脸检测MTCNN模型SDK未初始化，直接返回空");
        return NULL;
    }

    int tImageDateLen = env->GetArrayLength(imageDate_);
    if(imageChannel == tImageDateLen / imageWidth / imageHeight){
        LOGD("数据宽=%d,高=%d,通道=%d",imageWidth,imageHeight,imageChannel);
    }
    else{
        LOGD("数据长宽高通道不匹配，直接返回空");
        return NULL;
    }

    jbyte *imageDate = env->GetByteArrayElements(imageDate_, NULL);
    if (NULL == imageDate){
        LOGD("导入数据为空，直接返回空");
        env->ReleaseByteArrayElements(imageDate_, imageDate, 0);
        return NULL;
    }

    if(imageWidth<20||imageHeight<20){
        LOGD("导入数据的宽和高小于20，直接返回空");
        env->ReleaseByteArrayElements(imageDate_, imageDate, 0);
        return NULL;
    }

    //TODO 通道需测试
    if(3 == imageChannel || 4 == imageChannel){
        //图像通道数只能是3或4；
    }else{
        LOGD("图像通道数只能是3或4，直接返回空");
        env->ReleaseByteArrayElements(imageDate_, imageDate, 0);
        return NULL;
    }

    //int32_t minFaceSize=40;
    //mtcnn->SetMinFace(minFaceSize);

    unsigned char *faceImageCharDate = (unsigned char*)imageDate;
    ncnn::Mat ncnn_img;
    if(imageChannel==3) {
        ncnn_img = ncnn::Mat::from_pixels(faceImageCharDate, ncnn::Mat::PIXEL_BGR2RGB,
                                          imageWidth, imageHeight);
    }else{
        ncnn_img = ncnn::Mat::from_pixels(faceImageCharDate, ncnn::Mat::PIXEL_RGBA2RGB, imageWidth, imageHeight);
    }

    std::vector<Bbox> finalBbox;
    mDetect->detect(ncnn_img, finalBbox);

    int32_t num_face = static_cast<int32_t>(finalBbox.size());
    LOGD("检测到的人脸数目：%d\n", num_face);

    int out_size = 1+num_face*14;
    //  LOGD("内部人脸检测完成,开始导出数据");
    int *faceInfo = new int[out_size];
    faceInfo[0] = num_face;
    for(int i=0;i<num_face;i++){
        faceInfo[14*i+1] = finalBbox[i].x1;//left
        faceInfo[14*i+2] = finalBbox[i].y1;//top
        faceInfo[14*i+3] = finalBbox[i].x2;//right
        faceInfo[14*i+4] = finalBbox[i].y2;//bottom
        for (int j =0;j<10;j++){
            faceInfo[14*i+5+j]=static_cast<int>(finalBbox[i].ppoint[j]);
        }
    }

    jintArray tFaceInfo = env->NewIntArray(out_size);
    env->SetIntArrayRegion(tFaceInfo,0,out_size,faceInfo);
    //  LOGD("内部人脸检测完成,导出数据成功");
    delete[] faceInfo;
    env->ReleaseByteArrayElements(imageDate_, imageDate, 0);
    return tFaceInfo;
}

JNIEXPORT jintArray JNICALL
Java_com_aiface_1as_Face_MaxFaceDetect(JNIEnv *env, jobject instance, jbyteArray imageDate_,
                                      jint imageWidth, jint imageHeight, jint imageChannel) {
    //  LOGD("JNI开始检测人脸");
    if(!detection_sdk_init_ok){
        LOGD("人脸检测MTCNN模型SDK未初始化，直接返回空");
        return NULL;
    }

    int tImageDateLen = env->GetArrayLength(imageDate_);
    if(imageChannel == tImageDateLen / imageWidth / imageHeight){
        LOGD("数据宽=%d,高=%d,通道=%d",imageWidth,imageHeight,imageChannel);
    }
    else{
        LOGD("数据长宽高通道不匹配，直接返回空");
        return NULL;
    }

    jbyte *imageDate = env->GetByteArrayElements(imageDate_, NULL);
    if (NULL == imageDate){
        LOGD("导入数据为空，直接返回空");
        env->ReleaseByteArrayElements(imageDate_, imageDate, 0);
        return NULL;
    }

    if(imageWidth<20||imageHeight<20){
        LOGD("导入数据的宽和高小于20，直接返回空");
        env->ReleaseByteArrayElements(imageDate_, imageDate, 0);
        return NULL;
    }

    //TODO 通道需测试
    if(3 == imageChannel || 4 == imageChannel){
        //图像通道数只能是3或4；
    }else{
        LOGD("图像通道数只能是3或4，直接返回空");
        env->ReleaseByteArrayElements(imageDate_, imageDate, 0);
        return NULL;
    }

    //int32_t minFaceSize=40;
    //mtcnn->SetMinFace(minFaceSize);

    unsigned char *faceImageCharDate = (unsigned char*)imageDate;
    ncnn::Mat ncnn_img;
    if(imageChannel==3) {
        ncnn_img = ncnn::Mat::from_pixels(faceImageCharDate, ncnn::Mat::PIXEL_BGR2RGB,
                                          imageWidth, imageHeight);
    }else{
        ncnn_img = ncnn::Mat::from_pixels(faceImageCharDate, ncnn::Mat::PIXEL_RGBA2RGB, imageWidth, imageHeight);
    }

    std::vector<Bbox> finalBbox;
    mDetect->detectMaxFace(ncnn_img, finalBbox);

    int32_t num_face = static_cast<int32_t>(finalBbox.size());
    LOGD("检测到的人脸数目：%d\n", num_face);

    int out_size = 1+num_face*14;
    //  LOGD("内部人脸检测完成,开始导出数据");
    int *faceInfo = new int[out_size];
    faceInfo[0] = num_face;
    for(int i=0;i<num_face;i++){
        faceInfo[14*i+1] = finalBbox[i].x1;//left
        faceInfo[14*i+2] = finalBbox[i].y1;//top
        faceInfo[14*i+3] = finalBbox[i].x2;//right
        faceInfo[14*i+4] = finalBbox[i].y2;//bottom
        for (int j =0;j<10;j++){
            faceInfo[14*i+5+j]=static_cast<int>(finalBbox[i].ppoint[j]);
        }
    }

    jintArray tFaceInfo = env->NewIntArray(out_size);
    env->SetIntArrayRegion(tFaceInfo,0,out_size,faceInfo);
    //  LOGD("内部人脸检测完成,导出数据成功");
    delete[] faceInfo;
    env->ReleaseByteArrayElements(imageDate_, imageDate, 0);
    return tFaceInfo;
}



JNIEXPORT jboolean JNICALL
Java_com_aiface_1as_Face_SetMinFaceSize(JNIEnv *env, jobject instance, jint minSize) {
    if(!detection_sdk_init_ok){
        LOGD("人脸检测MTCNN模型SDK未初始化，直接返回");
        return false;
    }

    if(minSize<=20){
        minSize=20;
    }

    mDetect->SetMinFace(minSize);
    return true;
}


JNIEXPORT jboolean JNICALL
Java_com_aiface_1as_Face_SetThreadsNumber(JNIEnv *env, jobject instance, jint threadsNumber) {
    if(!detection_sdk_init_ok){
        LOGD("人脸检测MTCNN模型SDK未初始化，直接返回");
        return false;
    }

    if(threadsNumber!=1&&threadsNumber!=2&&threadsNumber!=4&&threadsNumber!=8){
        LOGD("线程只能设置1，2，4，8");
        return false;
    }

    retinafacenet->SetNumThreads(threadsNumber);
    return  true;
}


JNIEXPORT jboolean JNICALL
Java_com_aiface_1as_Face_SetTimeCount(JNIEnv *env, jobject instance, jint timeCount) {

    if(!detection_sdk_init_ok){
        LOGD("人脸检测MTCNN模型SDK未初始化，直接返回");
        return false;
    }

    mDetect->SetTimeCount(timeCount);
    return true;

}
}
// 添加人脸口罩分类函数接口  添加相应接口sdk


// 人脸识别
extern "C"
JNIEXPORT jdouble JNICALL
Java_com_aiface_1as_Face_FaceRecognize(JNIEnv *env, jobject instance,
                                      jbyteArray faceDate1_, jint w1, jint h1, jintArray landmarks1,
                                      jbyteArray faceDate2_, jint w2, jint h2, jintArray landmarks2) {
    double similar = 0;

    jbyte *faceDate1 = env->GetByteArrayElements(faceDate1_, NULL);
    jbyte *faceDate2 = env->GetByteArrayElements(faceDate2_, NULL);

    unsigned char *faceImageCharDate1 = (unsigned char *) faceDate1;
    unsigned char *faceImageCharDate2 = (unsigned char *) faceDate2;

    jint *mtcnn_landmarks1 = env->GetIntArrayElements(landmarks1, NULL);
    jint *mtcnn_landmarks2 = env->GetIntArrayElements(landmarks2, NULL);

    int *mtcnnLandmarks1 = (int *)mtcnn_landmarks1;
    int *mtcnnLandmarks2 = (int *)mtcnn_landmarks2;

    ncnn::Mat ncnn_img1 = ncnn::Mat::from_pixels(faceImageCharDate1, ncnn::Mat::PIXEL_RGBA2RGB, w1, h1);
    ncnn::Mat ncnn_img2 = ncnn::Mat::from_pixels(faceImageCharDate2, ncnn::Mat::PIXEL_RGBA2RGB, w2, h2);

    //人脸对齐
    ncnn::Mat det1 = mRecognize->preprocess(ncnn_img1, mtcnnLandmarks1);
    ncnn::Mat det2 = mRecognize->preprocess(ncnn_img2, mtcnnLandmarks2);

    std::vector<float> feature1, feature2;
    mRecognize->start(det1, feature1);
    mRecognize->start(det2, feature2);

    env->ReleaseByteArrayElements(faceDate1_, faceDate1, 0);
    env->ReleaseByteArrayElements(faceDate2_, faceDate2, 0);
    env->ReleaseIntArrayElements(landmarks1, mtcnn_landmarks1, 0);
    env->ReleaseIntArrayElements(landmarks2, mtcnn_landmarks2, 0);

    similar = calculSimilar(feature1, feature2, 1);
    return similar;
}

extern "C"
JNIEXPORT jdouble JNICALL
Java_com_aiface_1as_Face_maskfaceinfer(JNIEnv *env, jobject instance,
                                       jbyteArray faceDate1_) {


    jbyte *faceDate1 = env->GetByteArrayElements(faceDate1_, NULL);
    unsigned char *faceImageCharDate1 = (unsigned char *) faceDate1;
    ncnn::Mat ncnn_img1 = ncnn::Mat::from_pixels(faceImageCharDate1, ncnn::Mat::PIXEL_RGBA2RGB, 28, 28);
    double out=mMaskface->maskNet(ncnn_img1);
    env->ReleaseByteArrayElements(faceDate1_, faceDate1, 0);

    return out;
}