import numpy as np
import os
import math
from typing import Tuple, Sequence, Any

def sep():
    print("===========================================================================")

def save_arrays(fname: str, head_list: list, append: bool, *args: Tuple[Sequence[Any],...]):
    # fname (string): Name of file where the data is printed to. Has to include ending (e.g. ".txt")
    # head_list (List of strings): Array/List of names of the single arrays to build the headline of the document
    # args: all the arrays that want to be printed. Each array has its own column and is aligned with the corresponding values of the other arrays. Should have same dimension!
    head_string = ""
    if head_list is not None:
        for head in range(len(head_list)):
            head_string+=head_list[head]
            if head!=len(head_list)-1:
                head_string+="\t\t"
            else:
                head_string+="\n"
    assert len(args)>0, "You did not provide any data to save."     
    temp = np.expand_dims(args[0], -1)
    for i in range(1, len(args)):
        temp2 = np.expand_dims(args[i], -1)
        temp = np.concatenate((temp, temp2), -1)
    temp = temp.astype(str)
    if not os.path.exists(fname) or not append:
        with open(fname, "w") as f:
            f.write(head_string)
            for i in range(len(args[0])):
                string=""
                for j in range(len(args)):
                    string+=temp[i, j]
                    if j!=len(args)-1:
                        string+="\t\t"
                    else:
                        string+="\n"
                f.write(string)
    else:
        with open(fname, "a") as f:
            for i in range(len(args[0])):
                string=""
                for j in range(len(args)):
                    string+=temp[i, j]
                    if j!=len(args)-1:
                        string+="\t\t"
                    else:
                        string+="\n"
                f.write(string)

def round_to_n_of_err(x: float, xe: float, n: int = 2) -> Tuple[float, float]:
    """
    Round the value of x to n significant digits of its error xe. 
    """
    if xe == 0:
        return x, xe
    elif xe > np.fabs(x):
        return round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1)), round(xe, -int(math.floor(math.log10(abs(xe)))) + (n - 1))
    else:
        round_to = -int(math.floor(math.log10(abs(xe)))) + (n - 1)
        return round(x, round_to), round(xe, round_to)

