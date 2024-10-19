trailing_zeros=function(x){
    binary=intToBits(x)
    num_trailing_zeros=0
    for(i in 1:32){
        if(binary[i]==0) {
            num_trailing_zeros=num_trailing_zeros+1
        }
        else{
            break
        }
    }

    return(num_trailing_zeros)
}

hash_function=function(element){
    return(as.integer(element)%% 101)
}

fm=function(data_stream){
    max_num_trailing_zeros=0

    for(element in data_stream){
        hased_value=hash_function(element)
        zeros=trailing_zeros(hased_value)

        if(zeros>max_num_trailing_zeros){
            max_num_trailing_zeros=zeros
        }
    }
    return (2^max_num_trailing_zeros)
}

data_stream <- c(5, 15, 23, 23, 42, 42, 8, 99, 100, 5, 42)
distinct_elements=fm(data_stream)
print(distinct_elements)
