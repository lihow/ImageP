/***************************************************************************************************
*                    (c) Copyright 2012-? Institute of Information Engineering，Chinese Academy of Sciences
*                                       All Rights Reserved
*
*\File          shp_pic_interface.h
*\Description   水浒工程图文版式模块接口函数定义及规范
*\Log           2014.05.26    Ver 1.0    王旭林
*               创建文件。
***************************************************************************************************/
#ifndef __SH_PIC_INTERFACE_H__
#define __SH_PIC_INTERFACE_H__

#if defined WIN32 || defined _WIN32 || defined WINCE
#define WIN_SYSTEM		1
#ifdef OCRICDLL_EXPORTS
#define OCRICDLL_API __declspec(dllexport)
#else
#define OCRICDLL_API __declspec(dllimport)
#endif
#else
#define WIN_SYSTEM		0
#endif
//#include "stdafx.h"
#include <SDKDDKVer.h>  
#include <Windows.h>
#include <iostream>
#include <tchar.h> 

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cxcore.h>
#include <cv.h>
#include <highgui.h>
#include <io.h>
#include <time.h>
#include <hash_map>
#include <fstream>
#include <direct.h>
#include <sys/stat.h>

/************************************************************************/
/*                      数据类型定义                                    */
/************************************************************************/

/*日志格式数据长度定义*/
#define MAX_SECU_LENGTH 128                     /*MJ程度*/
#define MAX_URGENCY_LENGTH 128                  /*JinJi程度*/
#define MAX_FENHAO_LENGTH 128                   /*发文份号*/
#define MAX_DEP_LENGTH 256                      /*发文机构或者红头*/
#define MAX_TITLE_LENGTH 256                    /*发文标题*/
#define MAX_DELIVERY_TIME_LENGTH 128            /*发文日期*/
#define MAX_DELIVERY_NUM_LENGTH 128             /*发文编号*/
#define MAX_DELIVERY_CODE_LENGTH 128            /*发文代字*/
#define MAX_ADDTION_LENGTH 128                  /*附加信息*/
#define MAX_FILE_KEY_WORDS_LENGTH 4096          /*正文*/
#define MAX_KEY_WORD_NUM 128                    /*关键词最多数量*/

/*错误信息*/
enum OCR_StatusError
{
    OCR_SUCCESS											=0,		/*成功*/
    OCR_ERROR_OPEN_FEATURE_LIBRARY						=1,		/*打开特征库失败*/
    OCR_ERROR_OPEN_ENGLISH_CHAR_DICTIONARY				=2,		/*打开英文字典失败*/
    OCR_ERROR_OPEN_CHINESE_CHAR_DICTIONARY				=3,		/*打开汉字字典失败*/
    OCR_ERROR_OPEN_IMAGE_FILE							=7,		/*打开图像文件失败*/
    OCR_ERROR_RESIZE_IMAGE								=8,		/*缩放图像失败*/
};


/*四种检测模式*/
enum OCR_DetectMode
{
    KEYWORD_DETECT							= 0,    /*关键字检测*/
    LAYOUT_DETECT							= 1,    /*版式检测*/
    KEYWORD_LAYOUT_DETECT					= 2,    /*关键词检测和版式检测*/
    KEYWORD_LAYOUT_DETECT_FULLTEXT_EXTRACT	= 3	    /*关键词检测，版式检测，全文信息提取*/
};


 /*报警日志*/
 typedef  struct layout_log{
     int layout_cfg_id;					                        /*命中的版式配置ID*/
     int scan_file_type;					                    /*扫描件类型：-1为初始化，0标识非公文版式（命中了关键词或者mb），1标识疑似公文版式，2标识普通公文版式，3 SM公文版式*/
     char scan_file_secu[MAX_SECU_LENGTH];			            /*1表示内部文件，2表示MM，3表示JM，4表示J-M*/
     char scan_file_urgency[MAX_URGENCY_LENGTH];		        /*SM文件的紧急程度*/
     char scan_file_fenhao[MAX_FENHAO_LENGTH];		  	        /*SM文件的份号*/
     char scan_file_dep[MAX_DEP_LENGTH];			            /*文件所属部门的全称*/
     char scan_file_title[MAX_TITLE_LENGTH];			        /*标题*/
     char scan_file_delivery_time[MAX_DELIVERY_TIME_LENGTH];	/*发文时间*/
     char scan_file_delivery_num[MAX_DELIVERY_NUM_LENGTH];	    /*发文编号*/
     char scan_file_delivery_code[MAX_DELIVERY_CODE_LENGTH];	/*发文代字*/
     int pic_key_word[MAX_KEY_WORD_NUM];				        /*命中关键词ID，如命中1/3/5三个关键词，则数组赋值为{1,3,5,-1,-1,...,-1}*/
     char scan_file_addtion[MAX_ADDTION_LENGTH];		        /*附加信息*/
     int text_length;					                        /*正文长度*/
     char  scan_file_key_words[MAX_FILE_KEY_WORDS_LENGTH];	    /*正文内容*/
	 //OCR_LayoutLog *next;                                       /*多页tif时next为连接下页的节点*/
 }OCR_LayoutLog;

typedef struct analyze_result
{
    double	formatScore;	/*图片版式为公文的概率*/
    int		hasMask;		/*1为不含MM、JM、JueM等词语；2为MM；3为JM；4为jueM*/
    double  wordScore;		/*图片包含特定标识的概率*/
}OCR_AnalyzeResult;
/*模块句柄*/

typedef void* OCR_Handle;

/************************************************************************/
/*                      3个初始接口                                     */
/************************************************************************/

/**
	功能：版式匹配接口初始化,读取特征、英文、汉字3种字典，读取缺省配置文件
	参数：
		e[out]: 错误类型
	返回值：
	    模块句柄
**/

#if  WIN_SYSTEM
extern "C" OCRICDLL_API OCR_Handle OCR_InitLib(OCR_StatusError* e);
#else
extern "C" OCR_Handle OCR_InitLib(OCR_StatusError* e);
#endif
/**
	功能：版式匹配接口初始化，读取版式配置
	参数：
		handle[in]: 模块句柄
		layoutFileName[in]: 版式文件名称（相对路径+名称）若为nullptr，则采用默认配置里的版式文件名
		e[out]: 错误类型
	返回值：
	    true  初始化成功
		false 初始化失败
**/
#if  WIN_SYSTEM
extern "C" OCRICDLL_API bool OCR_LoadLayoutConf(OCR_Handle handle, const char* layoutFileName = NULL, OCR_StatusError* e = NULL);
#else
extern "C" bool OCR_LoadLayoutConf(OCR_Handle handle, const char* layoutFileName = NULL, OCR_StatusError* e = NULL);
#endif







/************************************************************************/
/*                           2个检测接口                                */
/************************************************************************/

/**
	功能：对指定的扫描件进行版式检测
	参数：
		handle[in]: 模块句柄
		filename[in]:扫描件名字（绝对路径+文件名）
		mode[in]:检测模式（4种）
		pciLayoutLog[out]:检测日志的结果，程序内部填写
		analyzeResult[out]:分析结果
		e[out]: 错误类型
	返回值：
		true  检测成功
		false 检测失败
**/
#if  WIN_SYSTEM
extern "C" OCRICDLL_API char* SINGLE_OCR(const char* fileName, OCR_StatusError* e);//单个图片文件接口
extern "C" OCRICDLL_API char* ViLab_OCR(const char *fileLoute, OCR_StatusError* e);//主接口（含pdf）
#endif


/************************************************************************/
/*               1个错误信息获取接口                                    */
/************************************************************************/

/**
	功能：错误信息的文字显示
	参数：
		e[in]: 错误信息代码
		
	返回值：
		errorInfo:错误信息
**/
#if  WIN_SYSTEM
extern "C" OCRICDLL_API char* OCR_GetError(OCR_StatusError* e);
#else
extern "C" char* OCR_GetError(OCR_StatusError* e);
#endif

/************************************************************************/
/*                      1个析构接口                                     */
/************************************************************************/

/**
	功能：析构接口动态申请的内存
	参数：
		handle[in]: 模块句柄
	返回值：
		true  释放成功
		false 释放失败
**/
#if  WIN_SYSTEM
extern "C" OCRICDLL_API bool OCR_DeleteLib(OCR_Handle handle);
#else
extern "C" bool OCR_DeleteLib(OCR_Handle handle);
#endif


/************************************************************************/
/*                       图文版式去噪                                     */
/************************************************************************/
/*
	功能：去除图文中的印章，离散点
	参数：
		char *fileLoute：图片文件路径
		OCR_StatuError e: 错误信息
	返回值：
		true  去噪成功
		false 去噪失败
*/
#if  WIN_SYSTEM
extern "C" OCRICDLL_API bool OCR_RemoveNoise(char* fileName, OCR_StatusError* e);
#else
extern "C"  bool OCR_RemoveNoise(char* fileName, OCR_StatusError* e);
#endif

/************************************************************************/
/*                       图文版式倾斜矫正                                     */
/************************************************************************/
/*
	功能：去除图文中的印章，离散点
	参数：
		char *fileLoute：图片文件路径
		OCR_StatuError e: 错误信息
	返回值：
		IplImage类型指针
*/
#if  WIN_SYSTEM
extern "C" OCRICDLL_API void OCR_SkewCorrection(char* fileName, OCR_StatusError* e);
#else
extern "C"  void OCR_SkewCorrection(char* fileName, OCR_StatusError* e);
#endif


/************************************************************************/
/*                       图文版式预处理                                     */
/************************************************************************/
/*
	功能：去除图文中的印章，离散点
	参数：
		char *fileName：图片文件路径
		char *dirr:图片保存目录，必须是已经存在的目录，否则会保存不成功
		OCR_StatuError e: 错误信息
	返回值：
		无
*/
#if  WIN_SYSTEM
extern "C" OCRICDLL_API void OCR_PicturePreprocess(char* fileName, char *dirr, OCR_StatusError* e);
#else
extern "C"  void OCR_PicturePreprocess(char* fileName, char *dirr, OCR_StatusError* e);
#endif

#if  WIN_SYSTEM
extern "C" OCRICDLL_API void OCR_QuaryImageProperty(char *fileName,  int attributes[][7], int n);
#else
extern "C"  void OCR_QuaryImageProperty(char *fileName,  int attributes[][7], int n);
#endif
#endif



