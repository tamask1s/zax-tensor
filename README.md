# zax-tensor
Yet another tensor library in C++. It allows direct access to its underlying data buffer, and serializes in JSON. Built on top of zax json parser, C++ structures having tensor members can also be JSON-serialized and deserialized, allowing one to save and load the state of a highly hierarchical object.

```cpp

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
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
    float_2d t_2d = R"([[81,90],
                       [0,2],
                       [-1,3]])";
    std::cout << t_2d << std::endl << std::endl;

    array_of_tensor_f32 aot;
    aot = R"([[81,90,0],
                 [0,2],
                 [-1,0,3,0,9]])";
    std::cout << aot << std::endl << std::endl;
    return 0;
}

```
#### Result:

```cpp

[
[81.000000,90.000000],
[0.000000,2.000000],
[-1.000000,3.000000]]

[[81.000000,90.000000,0.000000], [0.000000,2.000000], [-1.000000,0.000000,3.000000,0.000000,9.000000]]

```
