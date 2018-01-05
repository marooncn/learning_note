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
2.1> interactive programming using terminal
$ lua -i
You will come to interactive programming environment. You can use in-built 'print' function to print the content.
> print("Hello World")
Hello World！
2.2> .lua file
create a .lua file, such as hello.lua, the content is:
print("Hello World")
then you can excute the file
$ lua hello.lua
Hello World！
2.3> script
create a .lua file and specify the interpreter, such as hello.lua, the content is:
#!/usr/local/bin/lua
print("Hello World")
then you can excute the script
$ ./hello.lua
3 reference: http://www.lua.org/pil/contents.html
              http://www.runoob.com/lua/lua-tutorial.html
--]]
-- lua is a dynamic typing language, you don't need to define variable type before you use it.
a = 10
print(a)  -- 10
-- in lua language, all variables are global by default. To create a local variable, use 'local'.
local a = 5  -- create a local variable, when the block ends, it will be destroyed.
print(a)  -- 10
-- If you don't initianize a variable, its value is nil.
-- nil is a special variable and it means invalid in some way. In condition expressions, it is as false.
print(b)  -- nil
--------------------------------------------Data type---------------------------------------------------
-- Data types: nil, boolen, number, string, function, userdata, thread, table.
-- The nil type includes nil itself. You can use 'type' to look up the data type.
print(type(b)) -- nil
-- The boolen type includes ture and flase, just like other programming languages. 
-- The number type represents real (double-precision floating-point) numbers. So 'a' is not an integer data.
print(type(a)  -- number
print(type(2.2e-6) -- number
n1,n2 = 1,2    --n1=1, n2=2
n3,n4 = 3,4,5  --n1=3, n2=4
n5,n6 = 0      --n5=0, n6=nil. Pay attention the difference with other languages.
-- The string type uses single, double quotes or double square brackets to denote
string1 = 'this is string1'
string2 = "this is string2"
string3 = string1..' and '..string2  -- use '..' to connect string!
print(string3) --this is string1 and this is string2 
print(#string1) --15 , use '#' to get the string length.  
string4 = [[
cat
dog  ]]  -- it's convenient in some cases.
-- The table type implements associative arrays.
tab1 = {} --create a blank table.
tab1[1] = 'dog'
k = 'key'
tab1[k] = 'cat' 
print(tab1[k]) -- cat
-- Don't confuse a.x with a[x], a.x represents a['x'], that is, a table indexed by the string 'x'.
print(tab1.k)  -- nil , undefied tab1['k']  
-- when a numeric string invloves in the digital operations, lua will try convert it to number.    
print("2" + 6) -- 8.0
for x, y in pairs(tab1) do  -- generic for cycle, explained in control flow section
    print(x .. " : " .. y)  -- print the index and elements
end  
-- [[ Functions are first-class values in Lua. That means that functions can be stored in variables, passed as 
arguments to other functions, and returned as results. -- ]]
-- a simple example
function echo(x)  -- you can also use 'echo = function(x)' to create.
    print(x)
end
-- a  complex example  
function fcn2(n, fcn) -- you can also use 'fcn2 = function(n, fcn)' to create.
    n = fcn(n)
    if n == 0 then
        return 1
    else
        return n * fcn(n-1)
    end
end
fcn2_copy = fcn2  -- copy as a variable
print(fcn2_copy(-4, math.abs))  -- 12 , 'math.abs' is an in-built function to get the absolute value. 
-- For convenience, you can use anonymous function to pass function 
print(fcn2_copy(-4, function(x) return(math.abs(x)-1) end)) -- 3
-- Variable Arguments, just use '...' to denote variable arguments, which is similar to C.
function average(...)
   result = 0
   local arg={...}  
     -- local variable has its strength, for example less conflict(be destroyed after block) and faster operation.
   for i,v in ipairs(arg) do
      result = result + v
   end
   print("总共传入 " .. #arg .. " 个数")
   return result/#arg
end      
print("平均值为",average(10,5,3,4,5,6))    
-- The userdata type allows arbitrary C data to be stored in Lua variable
-- Coroutine a frequently used thread type in Lua.
-- A coroutine is similar to a thread, but there is only one running at any given time.

--------------------------------------------control flow---------------------------------------------------
-- if
-- In lua, except flase and lua, other variables are true including 0.
if(0)
then
    print("0 is true")
end      -- 0 is true
    
if(type(a) ~= number)
then
      print('a isn\'t the number type')   -- '\' is necessary to output character '
else
      if(a > 0)
      then
            print('a is positive')
      elseif(a < 0)
      then
            print('a is negative')
      else
            print('a is 0')
      end
end
    
-- while
a = 10
while( a < 20 )
do
   print("a 的值为:", a)
   a = a+1
end      -- a = 20, different with C which will be 10.
    
-- for 
for a=10,20,1  -- numberic cycle, a = start,end,step
do    
    print("a 的值为:", a)
end
days = {"Suanday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"}  
for i,v in ipairs(days)   -- generic cycle, i,v = index,value
do  
    print(v) 
end 
    
-- repeat ... until ...
a = 10
repeat
   print("a的值为:", a)
   a = a + 1
until( a > 20 )
-- break usage is the same as C.

-------------------------------------------- I/O ---------------------------------------------------
io.write("Hello")  --Hello
io.write("sin (3) = ", math.sin(3), "\n") -- sin (3) = 0.1411200080598672
-- 'io.write' is similar to 'print', but they are not the same.
io.write("hello", "Lua"); io.write("Hi", "\n")  -- helloLuaHi
print("hello", "Lua"); print("Hi")              -- hello    Lua
                                                -- Hi
-- file operation
-- simple model 
file = io.open("test.lua", "r+")        -- open test.lua with r+ format, format kinds are same as C.
io.input(file)                          -- set test.lua as the default input file
io.write("--  test.lua 文件末尾注释")     -- add content as the last line of test.lua
print(io.read())                        -- print the first line of test.lua
io.close(file)                          -- close test.lua
-- complete model (use the complete model for more control over I/O,)
file = io.open("test.lua", "r+")        -- open test.lua with r+ format
file:write("--  test.lua 文件末尾注释")  -- add content as the last line of test.lua
print(file:read())                     -- print the first line of test.lua
file:close()                           -- close test.lua
--[[ special arguments control for the read function
    "*all"	reads the whole file", "*line"	reads the next line, "*number"	reads a number
    num reads a string with up to num characters, fox example: ]] --
t = io.read("*all")                     -- read the whole file

--------------------------------------------Error Handling ---------------------------------------------------
-- pcall(protected call)
pcall(function(i) return 1/i end, 2)   -- true   0.5
pcall(function(i) return 1/i end, 0)   -- true   inf, so 1/0 is valid in Lua.
pcall(function(i) return 1/i end, 'a') -- false	stdin:1: attempt to perform arithmetic on a string value (local 'i')
-- xpcall (receive error handling argument to offer more information)
    -- use debug library including debug.debug and debug.traceback to get extra information and then pass to xpcall
xpcall(function(i) return 1/i end, function() print(debug.traceback()) end, 'a') 
xpcall(function(i) return 1/i end, function(err) print( "ERROR:", err ) end, 'a') 
