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
    tensor_f32 t_2d = R"([[81,90],
                          [0,2],
                          [-1,3]])";
    std::cout << t_2d;
    return 0;
}

```
#### Result:

```cpp

[
    [81.000000,90.000000],
    [0.000000,2.000000],
    [-1.000000,3.000000]]

```
# examples:
#### Example1 - create 3d tensor by shape:

##### Code:

```cpp

    ZaxJsonParser::set_indent(4);
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

    ZaxJsonParser::set_indent(4);
    tensor_f32 tensor_f32_({2,3,4});
    std::cout << tensor_f32_ << std::endl << std::endl;
    (*tensor_f32_.m_3d)[1][1][1] = 1;
    std::cout << tensor_f32_;

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

[
    [
        [0.000000,0.000000,0.000000,0.000000],
        [0.000000,0.000000,0.000000,0.000000],
        [0.000000,0.000000,0.000000,0.000000]],
    [
        [0.000000,0.000000,0.000000,0.000000],
        [0.000000,1.000000,0.000000,0.000000],
        [0.000000,0.000000,0.000000,0.000000]]]

```
#### Example4 - 4d data buffer access of a 4d tensor:

##### Code:

```cpp

    ZaxJsonParser::set_indent(4);
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

    ZaxJsonParser::set_indent(4);
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
#### Example6 -reshape a 4d tensor to a 4d tensor with different dimensions:

##### Code:

```cpp

    ZaxJsonParser::set_indent(4);
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
    std::cout << _4d << std::endl << std::endl;
    _4d.reshape(2,6,1,2);
    std::cout << _4d << std::endl << std::endl;

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

#### Example7 -reshape a 4d tensor to a 3d tensor with same number of elements:

##### Code:

```cpp

    ZaxJsonParser::set_indent(4);
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
    std::cout << _4d << std::endl << std::endl;
    _4d.reshape(2,6,1,2);
    std::cout << _4d << std::endl << std::endl;

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

[
    [
        [1.000000,2.000000,3.000000,4.000000,5.000000,6.000000],
        [7.000000,8.000000,9.000000,10.000000,11.000000,12.000000]],
    [
        [13.000000,14.000000,15.000000,16.000000,17.000000,18.000000],
        [19.000000,20.000000,21.000000,22.000000,23.000000,24.000000]]]
       
```
#### Example8 - create a 2d view of a 2d tensor with different shapes but same number of elements:

##### Code:

```cpp

    ZaxJsonParser::set_indent(4);
    tensor_f32 t_2d = R"([[0,1],
                          [2,3],
                          [4,5]])";
    tensor_f32 t_2d_view;
    t_2d_view.view({1, 6}, t_2d);
    (*t_2d_view.m_2d)[0][3] = 99;
    std::cout << t_2d << std::endl << std::endl;
    std::cout << t_2d_view;

```
##### Result:

```cpp

[
    [0.000000,1.000000],
    [2.000000,99.000000],
    [4.000000,5.000000]]

[
    [0.000000,1.000000,2.000000,99.000000,4.000000,5.000000]]

```
#### Example9 - create a 1d view of a 2d tensor with same number of elements:

##### Code:

```cpp

    ZaxJsonParser::set_indent(4);
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

    ZaxJsonParser::set_indent(4);
    tensor_f32 t_f32({1,3,3});
    std::cout << t_f32 << std::endl << std::endl;
    t_f32.squeeze(0);
    std::cout << t_f32 << std::endl << std::endl;
    t_f32.unsqueeze(0).unsqueeze(0);
    std::cout << t_f32 << std::endl << std::endl;

```
##### Result:

```cpp

[
    [
        [0.000000,0.000000,0.000000],
        [0.000000,0.000000,0.000000],
        [0.000000,0.000000,0.000000]]]

[
    [0.000000,0.000000,0.000000],
    [0.000000,0.000000,0.000000],
    [0.000000,0.000000,0.000000]]

[
    [
        [
            [0.000000,0.000000,0.000000],
            [0.000000,0.000000,0.000000],
            [0.000000,0.000000,0.000000]]]]

```
#### Example11 - array of tensors:

##### Code:

```cpp

    ZaxJsonParser::set_indent(4);
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
        tensor_f32 t_1d = R"([1,2,3])";
        tensor_f32 t_2d = R"([[1,2], [3,4], [5,6]])";
        ZAX_JSON_SERIALIZABLE(some_class, JSON_PROPERTY(t_1d), JSON_PROPERTY(t_2d))
    };

    ZaxJsonParser::set_indent(4);
    some_class some_obj;
    std::cout << some_obj;

```
##### Result:

```cpp

{
    "t_1d":[1.000000,2.000000,3.000000],
    "t_2d":[
        [1.000000,2.000000],
        [3.000000,4.000000],
        [5.000000,6.000000]]
}

```
