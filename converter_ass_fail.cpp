#include <iostream>
#include <fstream>
#include <string.h>
#include <time.h>
using namespace std;
int main (){
    fstream gogo ("ass.ass",ios::in);
    
    char h[5];
    float d;
    string lo;
    while (gogo>>lo){
       
         if(lo[0] =='0'&&lo[1] ==','&&lo[12]==","){
             
            cout <<lo<<"=========================="<<endl;
            h[0]=lo [7];
            h[1]=lo [8];
            h[2]=lo [9];
            h[3]=lo [10];
            h[4]=lo [11];
            d=atof (h);
            d=d+9;
            cout<<d<<"************"<<endl;

        }
        
    }
        gogo.close();

return 0;

}