--[[ 
1 install(Linux Terminal): 
$ curl -O http://www.lua.org/ftp/lua-5.3.4.tar.gz
$ tar zxf lua-5.3.4.tar.gz
$ cd lua-5.3.4
$ make linux test
$ make install
Then you can use lua.
2 Programming
To use lua to program, there are three approachs:
1.1> interactive programming using terminal
$ lua -i
You will come to interactive programming environment.
> print("Hello World")
Hello World！
1.2> .lua file
create a .lua file, such as hello.lua, the content is:
print("Hello World")
then you can excute the file
$ lua hello.lua
Hello World！
1.3> script
create a .lua file and specify the interpreter, such as hello.lua, the content is:
#!/usr/local/bin/lua
print("Hello World")
then you can excute the script
$ ./hello.lua
--]]



