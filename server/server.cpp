#include <stdio.h>
#include "xiApiPlusOcv.hpp"
#include <iostream>
#include <sys/socket.h> 
#include <arpa/inet.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <unistd.h> 
#include <string.h>
#include <stdint.h>
#include <cerrno>
#include <ctime>
#include <csignal>

using namespace cv;
using namespace std;

struct argumentStruct{
	int remoteSocket;
	xiAPIplusCameraOcv *camPtr;
};

static void *sendCameraBytes(void *ptr);
static bool transmitBytes(int socket, Mat& cv_mat_image, size_t imgSize);
static int remoteSocket;
static xiAPIplusCameraOcv cam;

//Shutdown function, called upon ctrl+c
static void closeDown(int sig);

int main(int argc, char** argv)
{   

  //--------------------------------------------------------
  //networking stuff: socket, bind, listen
  //--------------------------------------------------------
  int                 localSocket,
                      port = 4097;                               

  struct  sockaddr_in localAddr,
                      remoteAddr;
  pthread_t thread_id;
  
         
  int addrLen = sizeof(struct sockaddr_in);

     
  if ( (argc > 1) && (strcmp(argv[1],"-h") == 0) ) {
        std::cerr << "usage: ./cv_video_srv [port]\n" <<
                     "port           : socket port (4097 default)\n" <<
                     "capture device : (0 default)\n" << std::endl;

        exit(1);
  }

  if (argc == 2) port = atoi(argv[1]);

  localSocket = socket(AF_INET , SOCK_STREAM , 0);
  if (localSocket == -1){
       perror("socket() call failed!!");
  }    

  localAddr.sin_family = AF_INET;
  localAddr.sin_addr.s_addr = INADDR_ANY;
  localAddr.sin_port = htons( port );

  if( bind(localSocket,(struct sockaddr *)&localAddr , sizeof(localAddr)) < 0) {
       perror("Can't bind() socket");
       exit(1);
  }
  
  //Listening
  listen(localSocket , 3);
  
  std::cout <<  "Waiting for connections...\n"
            <<  "Server Port:" << port << std::endl;



signal(SIGINT, closeDown);

try
	{

		// Retrieving a handle to the camera device
		printf("Opening first camera...\n");
		cam.OpenFirst();
		
		// Set exposure
		cam.SetExposureTime(10000); //10000 us = 10 ms
		cam.SetBandwidthLimit(1000);		

	} catch(xiAPIplus_Exception& exp)
	{
		printf("Error:\n");
		exp.PrintError();
		cvWaitKey(2000);
		return 0;
	}

  while(1){
     
   remoteSocket = accept(localSocket, (struct sockaddr *)&remoteAddr, (socklen_t*)&addrLen);  
    //std::cout << remoteSocket<< "32"<< std::endl;
  if (remoteSocket < 0) {
      perror("accept failed!");
      exit(1);
  } 

  std::cout << "Connection accepted" << std::endl;
  argumentStruct argStruct = {remoteSocket, &cam};
  pthread_create(&thread_id,NULL,sendCameraBytes, &argStruct);
  pthread_join(thread_id,NULL);

  }
  while(1){}
  //pthread_join(thread_id,NULL);

  return 0;
}

static void *sendCameraBytes(void *ptr)
{
	argumentStruct* arguments = (argumentStruct*) ptr;
	int socket = arguments->remoteSocket;
	xiAPIplusCameraOcv cam = *(xiAPIplusCameraOcv*) arguments->camPtr;
	try
	{
		cam.StartAcquisition();
		int bytes = 0;
		printf("First pixel value \n");
		XI_IMG_FORMAT format = cam.GetImageDataFormat();
		while(1)
		{
      clock_t startTime = clock();
			Mat cv_mat_image = cam.GetNextImageOcvMat();
			if (format == XI_RAW16 || format == XI_MONO16){ 
				normalize(cv_mat_image, cv_mat_image, 0, 65536, NORM_MINMAX, -1, Mat()); // 0 - 65536, 16 bit unsigned integer range
			} 
      if ( ! cv_mat_image.isContinuous() ) { 
          cv_mat_image = cv_mat_image.clone();
      }  
      int imgSize = cv_mat_image.total() * cv_mat_image.elemSize();
        //cv::imshow("Image from camera",cv_mat_image);
				
        if(!transmitBytes(socket, cv_mat_image, imgSize)){
          cout<<"Error sending the packet."<<endl;
          cam.StopAcquisition();
          return NULL;
        }
        clock_t timeElapsed = (clock() - startTime) * 1000 / CLOCKS_PER_SEC;
        if(timeElapsed < 1000)
          cvWaitKey(1000 - timeElapsed);
		}
		
		printf("Done\n");
		
		//cvWaitKey(500);
	}
	catch(xiAPIplus_Exception& exp)
	{
		printf("Error:\n");
		exp.PrintError();
		cvWaitKey(2000);
	}
	return NULL;
}

static bool transmitBytes(int socket, Mat& cv_mat_image, size_t imgSize){
  uint32_t size = imgSize;
  assert(size == imgSize);
  int bytesSent = 0;
  clock_t timeElapsed;
  //Handshake with info about the size of the frame.
  do{
    clock_t startTime = clock();
    int retVal = send(socket, (const void *) &imgSize, sizeof(size)-bytesSent, 0);
    timeElapsed = (clock() - startTime) * 1000 / CLOCKS_PER_SEC;
    if(retVal == -1){
      cout<<strerror(errno)<<endl;
      return false;
    } else {
      bytesSent += retVal;
    }
  } while(bytesSent < sizeof(size));
  
  cout<<"Image size: "<<size<<", time spent on handshake: "<<timeElapsed<<" ms"<<endl;

  

  return true;
}

static void closeDown(int sig){
  close(remoteSocket);
  try{
   cam.Close();
  } catch(xiAPIplus_Exception e){
  }
  exit(0);
}