print("welcome to my computer quizz")

playing = input("do you want to play? ")

if playing != "yes":
    quit()

print("okay let's play :)! ")
score = 1

answer = input("what is your name bro? ")


if answer.lower() == "astrid":
    print("correct! ")
    score += 1
else:
    print("this computer is not for you leave of here my friend")

answer = input("what does RAM stand for? ")
if answer.lower()  == "random acces memory":
    print("correct! ")
    score += 1
else:
    print("incorrect")

answer = input("Did you teach python to somebody on this computer? ")
if answer.lower()  == "yes i did it":
    print("correct! ")
    score += 1
else:
    print("incorrect! ")

    print("you got " + str(score) + " questions correct! ")

    print("congratulation it's your computer")
