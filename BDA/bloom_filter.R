bloom_filter = rep(0, 10)

hash1 = function(element, size) {
    return ((as.integer(sum(as.numeric(charToRaw(element)))) %% size) + 1)
}

hash2 = function(element, size) {
    return (((as.integer(sum(as.numeric(charToRaw(element)))) * 31) %% size) + 1)
}

hash3 = function(element, size) {
    return (((as.integer(sum(as.numeric(charToRaw(element)))) * 101) %% size) + 1)
}

add_elements = function(bloom_filter, element) {
    size = length(bloom_filter)
    positions = c(hash1(element, size), hash2(element, size), hash3(element, size))

    bloom_filter[positions] = 1
    return(bloom_filter)
}

is_present = function(bloom_filter,element){
    size=length(bloom_filter)

    positions=c(hash1(element,size),hash2(element,size),hash3(element,size))

    return (all(bloom_filter[positions]==1))
}

bloom_filter = add_elements(bloom_filter, "hello")
bloom_filter = add_elements(bloom_filter, "vighnesh")


print(bloom_filter)
print(is_present(bloom_filter,"hello"))
print(is_present(bloom_filter,"no"))

