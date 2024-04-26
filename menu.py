#######################################
# Functions for validating user input #
#######################################

# Creates an input with prompt
# which checkes, if the input is empty
# If yes, the loop will continue until an input is entered
def input_empty(prompt):
    while(True):
        inp = input(prompt).strip()
        if(len(inp) == 0):
            print("No Input! Try again:")
        else:
            return inp   

# Creates an input with prompt
# which is checked, if the input is an integer number
# If not, the loop will continue until a valid number is entered
def input_int(prompt):
    while(True):
        nr = input(prompt)
        if not(check_int(nr)):
            print("Input is not an integer number! Try again:")
        else:
            return int(nr)  

# Creates an input for entering a prediction threshold
# If no value is entered, the standard threshold of 0.5 is used
def input_threshold(prompt):
    while(True):
        thr = input(prompt)
        if(thr == ""):
            print(f"Standard threshold value of 0.5 is used")
            return 0.5
        else:
            if not(check_nr(thr)):
                print("Input is not an number! Try again:")
            else:
                thr = float(thr)
                if(thr <= 0 or thr >= 1):
                    print("Threshold value must be > 0 and < 1! Try again:")
                else:
                    return thr    

####################################        
# Functions to check variable type #
####################################

# Check variable for number (int or float)
# Returns True if conversion was successful
# or False when the variable cannot be converted to a number
def check_nr(var):
    try:
        # Convert it into integer
        val = int(var)
        return True
    except ValueError:
        try:
            # Convert it into float
            val = float(var)
            return True
        except ValueError:
            return False

# Check variable for int
# Returns True if conversion was successful
# or False when the variable cannot be converted to an integer number
def check_int(var):
    try:
        val = int(var)
        return True
    except ValueError:
        return False
    
# Check variable for float
# Returns the variable as int if conversion was successful
# or False when the variable cannot be converted to an integer number
def check_float(var):
    try:
        # Convert it into integer
        val = int(var)
        return False
    except ValueError:
        try:
            # Convert it into float
            val = float(var)
            return True
        except ValueError:
            return False

##################
# Misc functions #
##################

# Function to exit any menue
def exit_menu(var, stop = "<exit>"):
    if(var == stop):
        print("Input canceled!")
        return True
    else:
        return False
    
# Prints a message to exit a menue with <exit>
def exit_menu_msg(stop = "<exit>"):
    print(f"> Enter {stop} to return to menue")
