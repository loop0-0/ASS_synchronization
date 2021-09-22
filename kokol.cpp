#include <iostream>
#include <fstream>
#include <string.h>
using namespace std;
int main (){
    fstream gogo ("ass.ass",ios::in);
    fstream lofi ("asolo.ass",ios:: app);
    int sqro=9;
    char h[5];
    char q[5];
    char h2[5];
    char q2[5];
    float s1,m1,s2,m2;
    string lo;
    while (gogo>>lo){
       
        if(lo =="Dialogue:"||lo =="Style:"){
            lofi<<endl<<lo<<" ";

        }
        else{
            
            if(lo[0] =='0'&&lo[1] ==','){
                //chater2
            h[0]=lo [4];
            h[1]=lo [5];
            q[0]=lo [7];
            q[1]=lo [8];
            q[2]=lo [9];
            q[3]=lo [10];
            q[4]=lo [11];
            m1=atof (h);
            s1=atof (q);
            s1=s1+sqro;
            if(s1>60){
                s1=s1-60;
                m1=m1+1;
            }
            string mm = to_string(m1);
            string ss =to_string(s1);
            lo [4]=mm[0];
            lo [5]=mm[1];
            lo [7]=ss[0];
            lo [8]=ss[1];
            lo [9]=ss[2];
            lo [10]=ss[3];
            lo [11]=ss[4];
            //chater1
            h2[0]=lo [15];
            h2[1]=lo [16];
            q2[0]=lo [18];
            q2[1]=lo [19];
            q2[2]=lo [20];
            q2[3]=lo [21];
            q2[4]=lo [22];
            m2=atof (h2);
            s2=atof (q2);
            s2=s2+sqro;
            if(s2>60){
                s2=s2-60;
                m2=m2+1;
            }
            string mm2 = to_string(m2);
            string ss2 =to_string(s2);
            lo [15]=mm2[0];
            lo [16]=mm2[1];
            lo [18]=ss2[0];
            lo [19]=ss2[1];
            lo [20]=ss2[2];
            lo [21]=ss2[3];
            lo [22]=ss2[4];

        lofi<<lo<<" ";}
        else {lofi<<lo<<" ";}
        }
    }
        gogo.close();
        lofi.close();

return 0;

}