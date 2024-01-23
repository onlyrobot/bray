@REM for dynamic link library
cl /c /EHsc /D_USRDLL -I"C:\Users\pengyao04\Downloads\boost_1_84_0" client.cpp
link /DLL /OUT:client.dll client.obj

@REM for static link library
cl /c /EHsc /MT -I"C:\Users\pengyao04\Downloads\boost_1_84_0" client.cpp
lib /nologo client.obj /NODEFAULTLIB