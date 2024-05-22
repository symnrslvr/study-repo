# Define the input and output file paths
input_file_path = 'image detection/origin/input.txt'
output_file_path = 'image detection/origin/output.txt'



# Initialize an empty list to store the converted data
converted_data = []

# Open the input file and read line by line
with open(input_file_path, 'r') as file:
    for line in file:
        # Strip any leading/trailing whitespace characters (e.g., newline characters)
        line = line.strip()
        
        # Replace '.' with ',' and ',' with ' ' at the appropriate positions
        parts = line.split(',')
        parts[0] = parts[0].replace('.', ',')
        converted_line = f"{parts[0]} {parts[1]},{parts[2]}"
        
        # Add the converted line to the list
        converted_data.append(converted_line)

# Print the converted data
print(converted_data)

# Optionally, write the converted data to an output file
with open(output_file_path, 'w') as file:
    for item in converted_data:
        file.write(f"{item}\n")