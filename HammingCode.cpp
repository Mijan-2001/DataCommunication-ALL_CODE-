#include<bits/stdc++.h>
using namespace std;
int main()
{

    int a[10],b[10];


    cout<<"Enter 4 bit:";
    cin>>a[3];
    cin>>a[5];
    cin>>a[6];
    cin>>a[7];

    a[1]=a[3]^a[5]^a[7];
    a[2]=a[3]^a[6]^a[7];
    a[4]=a[5]^a[6]^a[7];
    int temp[10];
    cout<<"Sending daga from sender to receiver ;";
    int j=1;
    for(int i=7;i>=1;i--)
    {
        cout<<a[i];
        temp[j++]=a[i];
    }
    cout<<endl;
    cout<<"Receiver code of len: ";
    for(int i=1;i<8;i++) cin>>b[i];
    bool ok=true;
    for(int i=1;i<8;i++)
    {

        if(temp[i]!=b[i])
        {
            ok==false;
            cout<<"Erroor"<<endl;
            cout<<"Error location :"<<i<<endl;
            return 0;
        }
    }

    cout<<"No error"<<endl;

    return 0;
}
