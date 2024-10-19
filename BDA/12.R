# Which function is used to concatenate text values in R. Write a script to concatenate text and
# numerical values in R.
# Text 1: Ram has scored
# Text 2: 89
# Text 3: marks
# Text 4: in Mathematics

text1="ram has scored"
text2=89
text3="marks"
text4="in mathematics"

with_spacing=paste(text1,text2,text3,text4)
without_spacing=paste0(text1,text2,text3,text4)

print(without_spacing)
# [1] "ram has scored89marksin mathematics"
print(with_spacing)
# [1] "ram has scored 89 marks in mathematics"
