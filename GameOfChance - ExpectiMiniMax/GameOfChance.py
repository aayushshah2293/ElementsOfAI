''' calculates the expected values of different rolls. 0 represents no dice rolled again, 1 represents 1 dice rolled again
2 represents pair of 2 dice rolled again and 3 represents all dice rolled again. The method returns the expected value of
all possible combinations and corresponding dices which were rolled again'''
def roll_dice(no_of_dice_to_roll, roll):
    sampespcae = [1, 2, 3, 4, 5, 6]
    final = []
    if no_of_dice_to_roll == 0:
        return [[calc_points(roll), '0']]

    elif no_of_dice_to_roll == 1:
        for i in range(3):
            expect = 0
            temp = roll.copy()
            temp.pop(i)
            for j in range(6):
                expect += calc_points(list(temp + [sampespcae[j]]))*(1/6)
            final += [[expect, str(i + 1)]]
        return final

    elif no_of_dice_to_roll == 2:
        for i in range(3):
            for j in range(i, 3):
                if i == j:
                    continue
                expect = 0
                temp = roll.copy()
                temp.pop(i)
                temp.pop(j-1)
                for k in range(6):
                    for l in range(6):
                        expect += calc_points(list(temp + [sampespcae[k]] + [sampespcae[l]])) * (1 / 36)

                final += [[expect, str(i+1) + " " + str(j + 1)]]
        return final

    elif no_of_dice_to_roll == 3:
        expect = 0
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    temp = [sampespcae[i], sampespcae[j], sampespcae[k]]
                    expect += calc_points(temp)*(1/216)
        return [[expect, "1 2 3"]]

'''calculates no of points of any given rolls'''
def calc_points(roll):
    if(roll[0] == roll[1] == roll[2]):
        return 25
    else:
        return sum(roll)

'''returns the most favourable move for a given sequence of rolls'''
def solve(roll):
    chance = []
    for i in range(4):
        chance += roll_dice(i, roll)
    return max(chance)


dice1 = int(input("What is Roll on Dice 1? "))  #input for dice 1
dice2 = int(input("What is Roll on Dice 2? "))  #input for dice 1
dice3 = int(input("What is Roll on Dice 3? "))  #input for dice 1
roll = [dice1,dice2,dice3]
move = solve(roll)
print("Roll Dice(s) No.: " + move[1])