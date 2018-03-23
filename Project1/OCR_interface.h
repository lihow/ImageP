/***************************************************************************************************
*                    (c) Copyright 2012-? Institute of Information Engineering��Chinese Academy of Sciences
*                                       All Rights Reserved
*
*\File          shp_pic_interface.h
*\Description   ˮ䰹���ͼ�İ�ʽģ��ӿں������弰�淶
*\Log           2014.05.26    Ver 1.0    ������
*               �����ļ���
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
/*                      �������Ͷ���                                    */
/************************************************************************/

/*��־��ʽ���ݳ��ȶ���*/
#define MAX_SECU_LENGTH 128                     /*MJ�̶�*/
#define MAX_URGENCY_LENGTH 128                  /*JinJi�̶�*/
#define MAX_FENHAO_LENGTH 128                   /*���ķݺ�*/
#define MAX_DEP_LENGTH 256                      /*���Ļ������ߺ�ͷ*/
#define MAX_TITLE_LENGTH 256                    /*���ı���*/
#define MAX_DELIVERY_TIME_LENGTH 128            /*��������*/
#define MAX_DELIVERY_NUM_LENGTH 128             /*���ı��*/
#define MAX_DELIVERY_CODE_LENGTH 128            /*���Ĵ���*/
#define MAX_ADDTION_LENGTH 128                  /*������Ϣ*/
#define MAX_FILE_KEY_WORDS_LENGTH 4096          /*����*/
#define MAX_KEY_WORD_NUM 128                    /*�ؼ����������*/

/*������Ϣ*/
enum OCR_StatusError
{
    OCR_SUCCESS											=0,		/*�ɹ�*/
    OCR_ERROR_OPEN_FEATURE_LIBRARY						=1,		/*��������ʧ��*/
    OCR_ERROR_OPEN_ENGLISH_CHAR_DICTIONARY				=2,		/*��Ӣ���ֵ�ʧ��*/
    OCR_ERROR_OPEN_CHINESE_CHAR_DICTIONARY				=3,		/*�򿪺����ֵ�ʧ��*/
    OCR_ERROR_OPEN_IMAGE_FILE							=7,		/*��ͼ���ļ�ʧ��*/
    OCR_ERROR_RESIZE_IMAGE								=8,		/*����ͼ��ʧ��*/
};


/*���ּ��ģʽ*/
enum OCR_DetectMode
{
    KEYWORD_DETECT							= 0,    /*�ؼ��ּ��*/
    LAYOUT_DETECT							= 1,    /*��ʽ���*/
    KEYWORD_LAYOUT_DETECT					= 2,    /*�ؼ��ʼ��Ͱ�ʽ���*/
    KEYWORD_LAYOUT_DETECT_FULLTEXT_EXTRACT	= 3	    /*�ؼ��ʼ�⣬��ʽ��⣬ȫ����Ϣ��ȡ*/
};


 /*������־*/
 typedef  struct layout_log{
     int layout_cfg_id;					                        /*���еİ�ʽ����ID*/
     int scan_file_type;					                    /*ɨ������ͣ�-1Ϊ��ʼ����0��ʶ�ǹ��İ�ʽ�������˹ؼ��ʻ���mb����1��ʶ���ƹ��İ�ʽ��2��ʶ��ͨ���İ�ʽ��3 SM���İ�ʽ*/
     char scan_file_secu[MAX_SECU_LENGTH];			            /*1��ʾ�ڲ��ļ���2��ʾMM��3��ʾJM��4��ʾJ-M*/
     char scan_file_urgency[MAX_URGENCY_LENGTH];		        /*SM�ļ��Ľ����̶�*/
     char scan_file_fenhao[MAX_FENHAO_LENGTH];		  	        /*SM�ļ��ķݺ�*/
     char scan_file_dep[MAX_DEP_LENGTH];			            /*�ļ��������ŵ�ȫ��*/
     char scan_file_title[MAX_TITLE_LENGTH];			        /*����*/
     char scan_file_delivery_time[MAX_DELIVERY_TIME_LENGTH];	/*����ʱ��*/
     char scan_file_delivery_num[MAX_DELIVERY_NUM_LENGTH];	    /*���ı��*/
     char scan_file_delivery_code[MAX_DELIVERY_CODE_LENGTH];	/*���Ĵ���*/
     int pic_key_word[MAX_KEY_WORD_NUM];				        /*���йؼ���ID��������1/3/5�����ؼ��ʣ������鸳ֵΪ{1,3,5,-1,-1,...,-1}*/
     char scan_file_addtion[MAX_ADDTION_LENGTH];		        /*������Ϣ*/
     int text_length;					                        /*���ĳ���*/
     char  scan_file_key_words[MAX_FILE_KEY_WORDS_LENGTH];	    /*��������*/
	 //OCR_LayoutLog *next;                                       /*��ҳtifʱnextΪ������ҳ�Ľڵ�*/
 }OCR_LayoutLog;

typedef struct analyze_result
{
    double	formatScore;	/*ͼƬ��ʽΪ���ĵĸ���*/
    int		hasMask;		/*1Ϊ����MM��JM��JueM�ȴ��2ΪMM��3ΪJM��4ΪjueM*/
    double  wordScore;		/*ͼƬ�����ض���ʶ�ĸ���*/
}OCR_AnalyzeResult;
/*ģ����*/

typedef void* OCR_Handle;

/************************************************************************/
/*                      3����ʼ�ӿ�                                     */
/************************************************************************/

/**
	���ܣ���ʽƥ��ӿڳ�ʼ��,��ȡ������Ӣ�ġ�����3���ֵ䣬��ȡȱʡ�����ļ�
	������
		e[out]: ��������
	����ֵ��
	    ģ����
**/

#if  WIN_SYSTEM
extern "C" OCRICDLL_API OCR_Handle OCR_InitLib(OCR_StatusError* e);
#else
extern "C" OCR_Handle OCR_InitLib(OCR_StatusError* e);
#endif
/**
	���ܣ���ʽƥ��ӿڳ�ʼ������ȡ��ʽ����
	������
		handle[in]: ģ����
		layoutFileName[in]: ��ʽ�ļ����ƣ����·��+���ƣ���Ϊnullptr�������Ĭ��������İ�ʽ�ļ���
		e[out]: ��������
	����ֵ��
	    true  ��ʼ���ɹ�
		false ��ʼ��ʧ��
**/
#if  WIN_SYSTEM
extern "C" OCRICDLL_API bool OCR_LoadLayoutConf(OCR_Handle handle, const char* layoutFileName = NULL, OCR_StatusError* e = NULL);
#else
extern "C" bool OCR_LoadLayoutConf(OCR_Handle handle, const char* layoutFileName = NULL, OCR_StatusError* e = NULL);
#endif







/************************************************************************/
/*                           2�����ӿ�                                */
/************************************************************************/

/**
	���ܣ���ָ����ɨ������а�ʽ���
	������
		handle[in]: ģ����
		filename[in]:ɨ������֣�����·��+�ļ�����
		mode[in]:���ģʽ��4�֣�
		pciLayoutLog[out]:�����־�Ľ���������ڲ���д
		analyzeResult[out]:�������
		e[out]: ��������
	����ֵ��
		true  ���ɹ�
		false ���ʧ��
**/
#if  WIN_SYSTEM
extern "C" OCRICDLL_API char* SINGLE_OCR(const char* fileName, OCR_StatusError* e);//����ͼƬ�ļ��ӿ�
extern "C" OCRICDLL_API char* ViLab_OCR(const char *fileLoute, OCR_StatusError* e);//���ӿڣ���pdf��
#endif


/************************************************************************/
/*               1��������Ϣ��ȡ�ӿ�                                    */
/************************************************************************/

/**
	���ܣ�������Ϣ��������ʾ
	������
		e[in]: ������Ϣ����
		
	����ֵ��
		errorInfo:������Ϣ
**/
#if  WIN_SYSTEM
extern "C" OCRICDLL_API char* OCR_GetError(OCR_StatusError* e);
#else
extern "C" char* OCR_GetError(OCR_StatusError* e);
#endif

/************************************************************************/
/*                      1�������ӿ�                                     */
/************************************************************************/

/**
	���ܣ������ӿڶ�̬������ڴ�
	������
		handle[in]: ģ����
	����ֵ��
		true  �ͷųɹ�
		false �ͷ�ʧ��
**/
#if  WIN_SYSTEM
extern "C" OCRICDLL_API bool OCR_DeleteLib(OCR_Handle handle);
#else
extern "C" bool OCR_DeleteLib(OCR_Handle handle);
#endif


/************************************************************************/
/*                       ͼ�İ�ʽȥ��                                     */
/************************************************************************/
/*
	���ܣ�ȥ��ͼ���е�ӡ�£���ɢ��
	������
		char *fileLoute��ͼƬ�ļ�·��
		OCR_StatuError e: ������Ϣ
	����ֵ��
		true  ȥ��ɹ�
		false ȥ��ʧ��
*/
#if  WIN_SYSTEM
extern "C" OCRICDLL_API bool OCR_RemoveNoise(char* fileName, OCR_StatusError* e);
#else
extern "C"  bool OCR_RemoveNoise(char* fileName, OCR_StatusError* e);
#endif

/************************************************************************/
/*                       ͼ�İ�ʽ��б����                                     */
/************************************************************************/
/*
	���ܣ�ȥ��ͼ���е�ӡ�£���ɢ��
	������
		char *fileLoute��ͼƬ�ļ�·��
		OCR_StatuError e: ������Ϣ
	����ֵ��
		IplImage����ָ��
*/
#if  WIN_SYSTEM
extern "C" OCRICDLL_API void OCR_SkewCorrection(char* fileName, OCR_StatusError* e);
#else
extern "C"  void OCR_SkewCorrection(char* fileName, OCR_StatusError* e);
#endif


/************************************************************************/
/*                       ͼ�İ�ʽԤ����                                     */
/************************************************************************/
/*
	���ܣ�ȥ��ͼ���е�ӡ�£���ɢ��
	������
		char *fileName��ͼƬ�ļ�·��
		char *dirr:ͼƬ����Ŀ¼���������Ѿ����ڵ�Ŀ¼������ᱣ�治�ɹ�
		OCR_StatuError e: ������Ϣ
	����ֵ��
		��
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



