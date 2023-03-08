# zax-tensor
Yet another tensor library in C++. It allows direct access to its underlying data buffer, and serializes in JSON. Built on top of zax json parser, C++ structures having tensor members can also be JSON-serialized and deserialized, allowing one to save and load the state of a highly hierarchical object.

```cpp

#include <iostream>
#include <typeinfo>
#include <string>
#include <string.h>
#include <map>
#include <vector>
#include <numeric>
#include "ZaxJsonParser.h"
#include "ZaxTensor.h"

int main()
{
    ZaxJsonParser::set_indent(4);

    tensor_i32 t_2d = R"([[81,90],
                          [0,2],
                          [-1,3]])";
    std::cout << t_2d;
    return 0;
}

```
#### Result:

```cpp

[
    [81,90],
    [0,2],
    [-1,3]]

```
# examples:
#### Example1 - create 3d tensor by shape:

##### Code:

```cpp

    tensor_f32 t_f32({2,3,4});
    std::cout << t_f32;

```
##### Result:

```cpp

[
    [
        [0.000000,0.000000,0.000000,0.000000],
        [0.000000,0.000000,0.000000,0.000000],
        [0.000000,0.000000,0.000000,0.000000]],
    [
        [0.000000,0.000000,0.000000,0.000000],
        [0.000000,0.000000,0.000000,0.000000],
        [0.000000,0.000000,0.000000,0.000000]]]

```

#### Example2 - display shape of a tensor:

##### Code:

```cpp

    tensor_f32 t_f32({2,3,4});
    std::vector<int> s = t_f32.shape();
    std::cout << s[0] << "/" << s[1] << "/" << s[2] << std::endl;
    std::cout << t_f32.shape_s();

```
##### Result:

```cpp

2/3/4
{2,3,4}

```

#### Example3 - accessing an element of a 3d tensor:

##### Code:

```cpp

    tensor_i32 tensor_i32_({2,3,4});
    std::cout << tensor_i32_ << std::endl << "------------------" << std::endl;
    (*tensor_i32_.m_3d)[0][0][1] = 5;
    std::cout << tensor_i32_;

```
##### Result:

```cpp

[
    [
        [0,0,0,0],
        [0,0,0,0],
        [0,0,0,0]],
    [
        [0,0,0,0],
        [0,0,0,0],
        [0,0,0,0]]]
------------------
[
    [
        [0,5,0,0],
        [0,0,0,0],
        [0,0,0,0]],
    [
        [0,0,0,0],
        [0,0,0,0],
        [0,0,0,0]]]

```
#### Example4 - 4d data buffer access of a 4d tensor:

##### Code:

```cpp

    tensor_f32 _4d({2,2,2,3});
    float**** b = _4d.data_4d();
    int c = 0;
    for (int i = 0; i < _4d.m_4d->d1; ++i)
        for (int j = 0; j < _4d.m_4d->d2; ++j)
            for (int k = 0; k < _4d.m_4d->d3; ++k)
                for (int l = 0; l < _4d.m_4d->d4; ++l)
                    b[i][j][k][l] = c++;
    std::cout << _4d;

```
##### Result:

```cpp

[
    [
        [
            [0.000000,1.000000,2.000000],
            [3.000000,4.000000,5.000000]],
        [
            [6.000000,7.000000,8.000000],
            [9.000000,10.000000,11.000000]]],
    [
        [
            [12.000000,13.000000,14.000000],
            [15.000000,16.000000,17.000000]],
        [
            [18.000000,19.000000,20.000000],
            [21.000000,22.000000,23.000000]]]]

```
#### Example5 - 1d data buffer access of a 4d tensor:

##### Code:

```cpp

    tensor_f32 _4d({2,2,2,3});
    float* b = _4d.data();
    for (int i = 0; i < _4d.numel(); ++i)
        b[i] = i;
    std::cout << _4d;

```
##### Result:

```cpp

[
    [
        [
            [0.000000,1.000000,2.000000],
            [3.000000,4.000000,5.000000]],
        [
            [6.000000,7.000000,8.000000],
            [9.000000,10.000000,11.000000]]],
    [
        [
            [12.000000,13.000000,14.000000],
            [15.000000,16.000000,17.000000]],
        [
            [18.000000,19.000000,20.000000],
            [21.000000,22.000000,23.000000]]]]

```
#### Example6 -reshape a 4d tensor to a 4d tensor with different dimensions (reshape will keep the original databuffer without reallocating it, only the tensor shell around it will be changed):

##### Code:

```cpp

    tensor_f32 _4d = R"([
    [
        [
            [1,2,3],
            [4,5,6]],
        [
            [7,8,9],
            [10,11,12]]],
    [
        [
            [13,14,15],
            [16,17,18]],
        [
            [19,20,21],
            [22,23,24]]]])";
    std::cout << _4d << std::endl << "------------------" << std::endl;
    _4d.reshape(2,6,1,2);
    std::cout << _4d;

```
##### Result:

```cpp

[
    [
        [
            [1.000000,2.000000,3.000000],
            [4.000000,5.000000,6.000000]],
        [
            [7.000000,8.000000,9.000000],
            [10.000000,11.000000,12.000000]]],
    [
        [
            [13.000000,14.000000,15.000000],
            [16.000000,17.000000,18.000000]],
        [
            [19.000000,20.000000,21.000000],
            [22.000000,23.000000,24.000000]]]]
------------------
[
    [
        [
            [1.000000,2.000000]],
        [
            [3.000000,4.000000]],
        [
            [5.000000,6.000000]],
        [
            [7.000000,8.000000]],
        [
            [9.000000,10.000000]],
        [
            [11.000000,12.000000]]],
    [
        [
            [13.000000,14.000000]],
        [
            [15.000000,16.000000]],
        [
            [17.000000,18.000000]],
        [
            [19.000000,20.000000]],
        [
            [21.000000,22.000000]],
        [
            [23.000000,24.000000]]]]
```

#### Example7 -reshape a 4d tensor to a 3d tensor with same number of elements (reshape will keep the original databuffer without reallocating it, only the tensor shell around it will be changed):

##### Code:

```cpp

    tensor_i32 _4d = R"([
    [
        [
            [1,2,3],
            [4,5,6]],
        [
            [7,8,9],
            [10,11,12]]],
    [
        [
            [13,14,15],
            [16,17,18]],
        [
            [19,20,21],
            [22,23,24]]]])";
    std::cout << _4d << std::endl << "------------------" << std::endl;
    _4d.reshape(2,2,6);
    std::cout << _4d;

```
##### Result:

```cpp

[
    [
        [
            [1,2,3],
            [4,5,6]],
        [
            [7,8,9],
            [10,11,12]]],
    [
        [
            [13,14,15],
            [16,17,18]],
        [
            [19,20,21],
            [22,23,24]]]]
------------------
[
    [
        [1,2,3,4,5,6],
        [7,8,9,10,11,12]],
    [
        [13,14,15,16,17,18],
        [19,20,21,22,23,24]]]
       
```
#### Example8 - create a 2d view of a 2d tensor with different shapes but same number of elements:

##### Code:

```cpp

    tensor_i32 t_2d = R"([[0,1],
                          [2,3],
                          [4,5],
                          [6,7]])";
    tensor_i32 t_2d_view;
    t_2d_view.view({2, 4}, t_2d);
    (*t_2d_view.m_2d)[0][3] = 99;
    std::cout << t_2d << std::endl << "------------------" << std::endl;
    std::cout << t_2d_view;

```
##### Result:

```cpp

[
    [0,1],
    [2,99],
    [4,5],
    [6,7]]
------------------
[
    [0,1,2,99],
    [4,5,6,7]]

```
#### Example9 - create a 1d view of a 2d tensor with same number of elements:

##### Code:

```cpp

    tensor_f32 t_2d = R"([[0,1],
                          [2,3],
                          [4,5]])";
    tensor_f32 t_2d_view;
    t_2d_view.view({6}, t_2d);
    (*t_2d_view.m_2d)[3] = 99;
    std::cout << t_2d << std::endl << std::endl;
    std::cout << t_2d_view;

```
##### Result:

```cpp

[
    [0.000000,1.000000],
    [2.000000,99.000000],
    [4.000000,5.000000]]

[0.000000,1.000000,2.000000,99.000000,4.000000,5.000000]

```
#### Example10 - squeeze and unsqueeze:

##### Code:

```cpp

    tensor_i32 t_i32({1,3,3});
    std::cout << t_i32 << std::endl << "------------------" << std::endl;
    t_i32.squeeze(0);
    std::cout << t_i32 << std::endl << "------------------" << std::endl;
    t_i32.unsqueeze(0).unsqueeze(0);
    std::cout << t_i32;

```
##### Result:

```cpp

[
    [
        [0,0,0],
        [0,0,0],
        [0,0,0]]]
------------------
[
    [0,0,0],
    [0,0,0],
    [0,0,0]]
------------------
[
    [
        [
            [0,0,0],
            [0,0,0],
            [0,0,0]]]]

```
#### Example11 - array of tensors:

##### Code:

```cpp

    array_of_tensor_f32 aot = R"([[81,90,0],
                                  [[0,2],[5,6],[7,8]],
                                  [-1,0,3,0,9]])";
    std::cout << aot;

```
##### Result:

```cpp

[[81.000000,90.000000,0.000000],
    [
        [0.000000,2.000000],
        [5.000000,6.000000],
        [7.000000,8.000000]],
    [-1.000000,0.000000,3.000000,0.000000,9.000000]]

```
#### Example12 - serialization of a class containing tensors:

##### Code:

```cpp

    struct some_class
    {
        tensor_i32 t_1d = R"([1,2,3])";
        tensor_i32 t_2d = R"([[1,2], [3,4], [5,6]])";
        ZAX_JSON_SERIALIZABLE(some_class, JSON_PROPERTY(t_1d), JSON_PROPERTY(t_2d))
    };

    some_class some_obj;
    std::cout << some_obj;

```
##### Result:

```cpp

{
    "t_1d":[1,2,3],
    "t_2d":[
        [1,2],
        [3,4],
        [5,6]]
}

```
#### Example13 - Arduino example:

##### Code:
```cpp
#include <cstring>
#include <stdio.h>
#include <string>
#include <map>
#include <random>
#include <vector>
#include <algorithm>
#include <typeinfo>
#include <math.h>
#include "src/ZaxJsonParser.h"
#include "src/ZaxTensor.h"

struct some_class
{
    int x = 9;
    std::string name = "some name";
    ZAX_JSON_SERIALIZABLE(some_class, JSON_PROPERTY(x), JSON_PROPERTY(name))
};

void setup(void)
{
    Serial.begin(115200);
    delay(1000);
    ZaxJsonParser::set_initial_alloc_size(5000);

    Serial.println();Serial.println("Initializing and printing an object of 'some_class':");
    some_class some_obj = R"({"x": 7, "name": "new name"})";
    std::string some_json = some_obj;
    Serial.println(some_json.c_str());

    Serial.println();Serial.println("Using 'zax_to_json()' as serialization method:");
    some_json = some_obj.zax_to_json();
    Serial.println(some_json.c_str());
    
    Serial.println();Serial.println("Printing JSON in a 'char some_json_cstr[200]':");
    char some_json_cstr[200];
    some_obj.zax_to_json(some_json_cstr, 200);
    Serial.println(some_json_cstr);

    Serial.println();Serial.println("Printing an unsigned char matrix (value of '-1' is displad as '255'):");
    tensor_ui8 t_2d_i8 = R"([[8,90],
                             [0,2],
                             [-1,3]])";
    // some_json = t_2d_i8; - this won't work for now
    std::string some_json2 = t_2d_i8;
    Serial.println(some_json2.c_str());
    
    Serial.println();Serial.println("Printing an int matrix:");
    tensor_i32 t_2d_i32 = R"([[32,90],
                              [0,2],
                              [-1,3]])";
    std::string some_json3 = t_2d_i32;
    Serial.println(some_json3.c_str());
    
    Serial.println();Serial.println("Printing a float matrix:");
    tensor_f32 t_2d_f32 = R"([[32,90],
                              [0,2],
                              [-1,3]])";
    std::string some_json4 = t_2d_f32;
    Serial.println(some_json4.c_str());
    
    Serial.println();Serial.println("Creating mx using external buff. Filling it, and setting element values in different ways:");
    tensor_ui16 t_2d_ui16;
    uint16_t* backBuffer16 = (uint16_t*) malloc(4 * 8 * 2);
    t_2d_ui16.create({4, 8}, backBuffer16, true);
    t_2d_ui16.fill(0);
    uint16_t** screen = t_2d_ui16.data_2d();
    screen[0][0] = 12;
    backBuffer16[1 * 8 + 1] = 34;
    t_2d_ui16(2, 2) = 56;
    (*t_2d_ui16.m_2d)[3][3] = 78;
    std::string t_2d_ui16_str = t_2d_ui16;
    Serial.println(t_2d_ui16_str.c_str());
}

void loop(void)
{}

```
##### Result:

```cpp
Initializing and printing an object of 'some_class':
{"x":7, "name":"new name"}

Using 'zax_to_json()' as serialization method:
{"x":7, "name":"new name"}

Printing JSON in a 'char some_json_cstr[200]':
{"x":7, "name":"new name"}

Printing an unsigned char matrix (value of '-1' is displad as '255'):
[
[8,90],
[0,2],
[255,3]]

Printing an int matrix:
[
[32,90],
[0,2],
[-1,3]]

Printing a float matrix:
[
[32.000000,90.000000],
[0.000000,2.000000],
[-1.000000,3.000000]]

Creating mx using external buff. Filling it, and setting element values in different ways:
[
[12,0,0,0,0,0,0,0],
[0,34,0,0,0,0,0,0],
[0,0,56,0,0,0,0,0],
[0,0,0,78,0,0,0,0]]
```
