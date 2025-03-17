import time
import numpy as np
import os
import math
import h5py

from typing import Union, Sequence, Tuple, Any


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
                    
def round_to_n(x: float, n: int) -> float:
    """
    Round the input value x to the n-th figure. 
    """
    if x == 0:
        return 0
    else:
        return round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1))
    
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
    
def get_parentheses_error_notation(value: float, error: float) -> str:
    """
    Takes a value and its error and returns a string with their parentheses notation with two significant digits. 
    E.g. for value = 0.1234 and error = 0.056 the output is "0.123(56)"
    """
    value, error = float(value), float(error)
    sv, se = str(value), str(error)
    minus = False
    if sv[0]=="-":
        minus = True
        sv = sv[1:]
    div, die = sv.find("."), se.find(".")
    di = max(div, die)
    v_bd, v_ad = sv.split(".")
    e_bd, e_ad = se.split(".")
    lbd = max(len(v_bd), len(e_bd))
    lad = max(len(v_ad), len(e_ad))+1 
    while len(v_bd)<lbd:
        v_bd = "0"+v_bd
    while len(e_bd)<lbd:
        e_bd = "0"+e_bd
    while len(v_ad)<lad:
        v_ad += "0"
    while len(e_ad)<lad:
        e_ad += "0"    
    pad_v = v_bd + v_ad
    pad_e = e_bd + e_ad
    for i in range(len(pad_e)):
        if pad_e[i]!="0":
            break
    e_digits = pad_e[i:i+2]
    val_digits = pad_v[:i+2]
    result = val_digits + "(" + e_digits + ")"
    if len(val_digits) < len(v_bd):
        result += "e"+str(len(v_bd)-len(val_digits)) 
    elif len(val_digits) > len(v_bd): 
        result = result[:di]+"."+result[di:]
    if minus:
        result = "-" + result
    return result
   
def sep():
    print("===========================================================================================")

def convert_from_scientific_notation(number_string: str) -> str:
    """
    This function converts a string of a float that is provided in scientific notation to the "standard" decimal notation. 
        args: number_string (string) : string of float number
    """
    if "e" in number_string: 
        e_ind = number_string.find("e")
        power_str = number_string[e_ind+2:]
        sign = number_string[e_ind+1]
        number_string = number_string[:e_ind]
        if sign == "+":
            if "." in number_string:
                dot_ind = number_string.find(".")
                number_string = number_string.replace(".", "")
                number_string = number_string + (-len(number_string) + dot_ind + int(power_str))*"0"+".0"
            else:
                number_string = number_string + int(power_str)*"0"+".0"
        elif sign == "-":
            if "." in  number_string:
                dot_ind = number_string.find(".")
                number_string = number_string.replace(".", "")
                number_string = "0."+ (int(power_str) - 1)*"0" + number_string
            else:
                number_string = "0."+ (int(power_str)-1)*"0" + number_string
        else:
            print("Sign in exponent neither + nor -. This is not implemented. ")
            pass
    return number_string

def sort_arrays_according_to_array(array: np.ndarray, *args: Tuple[np.ndarray,...]) -> Tuple[np.ndarray,...]:
    """
    array: the array to be sorted
    args: any number of arrays with same length as array that are sorted with the same indices as array
    """
    sort_ind = np.argsort(array)
    sorted_array = array[sort_ind]
    additional_sorted_arrays = []
    for i in range(len(args)):
        additional_sorted_arrays.append(args[i][sort_ind])
    return sorted_array, *additional_sorted_arrays

def delete_indices_from_arrays(indices: Union[list, tuple], *args: Tuple[np.ndarray,...]) -> Tuple[Union[list,tuple,np.ndarray],...]:
    """
    indices: the indices to be deleted
    args: any number of arrays with length smaller than largest value of indices
    """
    modified_arrays = []
    for i in range(len(args)):
        modified_arrays.append(np.delete(args[i], indices))
    return indices, *modified_arrays

# def get_NSL_lattices_from_h5(path, Nconf_max = 1e12):
#     t1 = time.time()
#     with h5py.File(path, "r") as f:
#         name = str(*f.keys())
#         direc = name+"/markovChain/0/phi"
#         dset_tmp = f[direc]
#         lat_tmp = np.real(dset_tmp[()])
#         lat_shape = lat_tmp.shape
#         gdir = name+"/markovChain"
#         Nconf = len(f[gdir].keys())
#         # print(Nconf)
#         actual_Nconf = min(Nconf, int(Nconf_max))
#         shape = (actual_Nconf, *lat_shape)
#         full_data = np.zeros(shape)
#         t2 = time.time()
#         for i in range(actual_Nconf):
#             direc = name+f"/markovChain/{i}/phi"
#             dset_tmp = f[direc]
#             # data = np.real(dset_tmp[()])
#             # full_data[i] = data
#             full_data[i] = np.real(dset_tmp[()])
#     return full_data


def get_NSL_lattices_from_h5(path: str, Nconf: int) -> np.ndarray:
    t1 = time.time()
    with h5py.File(path, "r") as f:
        name = str(*f.keys())
        direc = name+"/markovChain/0/phi"
        lat_tmp = np.real(f[direc][()])
        lat_shape = lat_tmp.shape
        full_data = np.zeros((Nconf, *lat_shape))
        t2 = time.time()
        for i in range(Nconf):
            direc = name+f"/markovChain/{i}/phi"
            dset_tmp = f[direc]
            # data = np.real(dset_tmp[()])
            # full_data[i] = data
            full_data[i] = np.real(dset_tmp[:])
    return full_data


def get_NSL_correlators_from_h5(path: str, Nt: int, Nconf_max: int = int(1e5)) -> np.ndarray:
    t1 = time.time()
    shape = (Nt, 2, 2)
    full_corr = np.zeros((Nconf_max, Nt, 2, 2), dtype = np.complex128)
    with h5py.File(path, "r") as f:
        name = str(*f.keys())
        Nconf = len(f[name+"/markovChain"])
        for i in range(Nconf_max):
            corr = np.reshape(f[name+f"/markovChain/{i}/correlators/single/particle/k0"][:], shape)
            full_corr[i, :, :, :] = corr
    print(f"Loaded data in {round(time.time()-t1, 2)}s")
    return full_corr
