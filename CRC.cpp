#include<bits/stdc++.h>
using namespace std;

int main()
{
    string msg,crc,encoded="";
    cout<<"Enter the message ";
    getline(cin,msg);
    cout<<"Enter crc generator : ";
    getline(cin,crc);

    int m=msg.length(),n=crc.length();

    encoded += msg;
    for(int i=1; i<=n-1; i++) encoded += '0';

    for(int i=0; i<=encoded.length()-n;)
    {
        for(int j=0; j<n; j++)
        {
            encoded[i+j]=encoded[i+j]==crc[j]?'0':'1';

        }
        for(; i<encoded.length() && encoded[i]!='1'; i++);
    }
    cout<<"The message sent from sender to receiver : "<<msg+encoded.substr(encoded.length()-n+1)<<endl;

    ////////////////////////////


    cout<<"Enter the message ";
    getline(cin,encoded);

    n=crc.length();

    // encoded += msg;
    //for(int i=1;i<=n-1;i++) encoded += '0';

    for(int i=0; i<encoded.length()-n;)
    {
        for(int j=0; j<n; j++)
        {
            encoded[i+j]=encoded[i+j]==crc[j]?'0':'1';

        }
        for(; i<encoded.length() && encoded[i]!='1'; i++);
    }

    for(char i:encoded.substr(encoded.length()-n))
    {
        if( i!= '0')
        {
            cout<<"Error is detected"<<endl;
            return 0;
        }
    }
    cout<<"NO error"<<endl;

}
