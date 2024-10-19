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
