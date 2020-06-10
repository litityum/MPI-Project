#Ömer Yılmaz
#Compiling
#Working
#Periodic and Checkered
from mpi4py import MPI
import numpy as np
import math
import sys

#Initilazing mpi and getting rank
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
#Getting necessary variables
Grid_Size = 360
worker_count = comm.Get_size()-1
w_per_dimension = int(math.sqrt(worker_count))
Rank_size = Grid_Size//w_per_dimension
time = int(sys.argv[3])
#method to get rank from col and row
def to_rank(row, col):
    row = row % w_per_dimension
    col = col % w_per_dimension
    return row * w_per_dimension + col + 1
#method to get row and col from rank
def from_rank(rank):
    row = (rank - 1) // w_per_dimension
    col = (rank - 1) % w_per_dimension
    return (row, col)
#calculating the next value of a square according to game of life rules
def next_sqr(center, sum):
    if center == 0:
        if sum == 3:
            return 1
        else:
            return 0
    else:
        if sum < 2 or sum > 3:
            return 0
        else:
            return 1      
#a method to calculate the next situation of grid of a worker
def next_step(upleft, up, upright, downleft, down, downright, left, right, data):
    #New values will be here
    new_data = np.zeros((Rank_size,Rank_size), dtype=int)
    #Concatenating the matrices
    top = np.concatenate((upleft, up, upright), axis=1)
    middle = np.concatenate((left, data, right), axis=1)
    bot = np.concatenate((downleft, down, downright), axis=1)
    #The extended matrix
    yeni = np.concatenate((top, middle, bot), axis=0)
    #Calculating all the next values of the matrix
    for i in range(Rank_size):
        new_data[i] = [next_sqr(data[i][j], yeni[i][j]+ yeni[i][j+1]+ yeni[i][j+2]+ yeni[i+1][j]+ yeni[i+1][j+2]+ yeni[i+2][j]+ yeni[i+2][j+1]+ yeni[i+2][j+2]) for j in range(Rank_size)]
    return new_data
    
#Beginning of the process
#Master organizing the input
if rank == 0:
    #Getting the arguments
    input = sys.argv[1]
    output = sys.argv[2]
    #Loading the input
    grid = np.loadtxt(input, dtype=int)
    #Partition of the input and sending it to workers
    for i in range(1, worker_count + 1):
            row = from_rank(i)[0]
            col = from_rank(i)[1]
            data = np.copy(grid[row*Rank_size : (row+1)*Rank_size, col*Rank_size : (col+1)*Rank_size])
            comm.Send(data, dest = i)
#If not master than calculate the row and col, receive the matrix
else:
    row = from_rank(rank)[0]
    col = from_rank(rank)[1]
    data = np.zeros((Rank_size, Rank_size), dtype=int)
    comm.Recv(data, source = 0)
#Let the process begins
for t in range(time):
    #There is nothing to do for master here
    if rank != 0:
        #Creating empty arrays to be received from other workers
        right = np.zeros((Rank_size, 1), dtype=int)
        left = np.zeros((Rank_size, 1), dtype=int)
        down = np.zeros((1, Rank_size), dtype=int)
        up = np.zeros((1, Rank_size), dtype=int)
        upright = np.zeros((1,1), dtype=int)
        upleft = np.zeros((1,1), dtype=int)
        downright = np.zeros((1,1), dtype=int)
        downleft = np.zeros((1,1), dtype=int)
        #If it is a white square in Chess, it sends first
        if (row + col) % 2 == 0:
            #Sending left, right, up, down respectively
            comm.Send(np.copy(data[:, :1]), dest = to_rank(row, col - 1))
            comm.Send(np.copy(data[:, -1:]), dest = to_rank(row, col + 1))
            comm.Send(np.copy(data[:1, :]), dest = to_rank(row - 1, col))
            comm.Send(np.copy(data[-1:, :]), dest = to_rank(row + 1, col))
            #Receiving right, left, down, up respectively
            #Which are the opposite of the sending order in order to prevent Deadlocks            
            comm.Recv(right, source = to_rank(row, col + 1))
            comm.Recv(left, source = to_rank(row, col - 1))
            comm.Recv(down, source = to_rank(row + 1, col))
            comm.Recv(up, source = to_rank(row - 1, col))
        #If it is a black square
        else:
            #Receiving right, left, down, up respectively
            #Which are the opposite of the sending order in order to prevent Deadlocks            
            comm.Recv(right, source = to_rank(row, col + 1))
            comm.Recv(left, source = to_rank(row, col - 1))
            comm.Recv(down, source = to_rank(row + 1, col))
            comm.Recv(up, source = to_rank(row - 1, col))
            #Sending left, right, up, down respectively
            comm.Send(np.copy(data[:, :1]), dest = to_rank(row, col - 1))
            comm.Send(np.copy(data[:, -1:]), dest = to_rank(row, col + 1))
            comm.Send(np.copy(data[:1, :]), dest = to_rank(row - 1, col))
            comm.Send(np.copy(data[-1:, :]), dest = to_rank(row + 1, col))
        #It is muffin time!
        #Sending the corners according to col is odd or not
        if col % 2 == 0:
            #Sending upleft, upright, downleft, downright respectively
            comm.Send(np.copy(data[:1, :1]), dest = to_rank(row - 1, col - 1))
            comm.Send(np.copy(data[:1, -1:]), dest = to_rank(row - 1, col + 1))
            comm.Send(np.copy(data[-1:, :1]), dest = to_rank(row + 1, col - 1))
            comm.Send(np.copy(data[-1:, -1:]), dest = to_rank(row + 1, col + 1))
            #Receiving upleft, upright, downleft, downright respectively
            #Which are the opposite of the sending order in order to prevent Deadlocks           
            comm.Recv(downright, source = to_rank(row + 1, col + 1))
            comm.Recv(downleft, source = to_rank(row + 1, col - 1))
            comm.Recv(upright, source = to_rank(row - 1, col + 1))
            comm.Recv(upleft, source = to_rank(row - 1, col - 1))   
        else:
            #Receiving upleft, upright, downleft, downright respectively
            #Which are the opposite of the sending order in order to prevent Deadlocks again
            comm.Recv(downright, source = to_rank(row + 1, col + 1))
            comm.Recv(downleft, source = to_rank(row + 1, col - 1))
            comm.Recv(upright, source = to_rank(row - 1, col + 1))
            comm.Recv(upleft, source = to_rank(row - 1, col - 1))
            #Sending upleft, upright, downleft, downright respectively
            comm.Send(np.copy(data[:1, :1]), dest = to_rank(row - 1, col - 1))
            comm.Send(np.copy(data[:1, -1:]), dest = to_rank(row - 1, col + 1))
            comm.Send(np.copy(data[-1:, :1]), dest = to_rank(row + 1, col - 1))
            comm.Send(np.copy(data[-1:, -1:]), dest = to_rank(row + 1, col + 1))
        #Join all the info gathered from other workers and calculate the new matrix
        data = next_step(upleft, up, upright, downleft, down, downright, left, right, data)
#Lets finish the process
#Master collecting the data and saving it
if rank == 0:
    new_grid = np.zeros((Grid_Size,Grid_Size), dtype=int)
    #getting data
    for i in range(1, worker_count + 1):
        row = from_rank(i)[0]
        col = from_rank(i)[1]
        data = np.zeros((Rank_size,Rank_size), dtype=int)
        comm.Recv(data, source = i)
        new_grid[row*Rank_size : (row+1)*Rank_size, col*Rank_size : (col+1)*Rank_size] = data
    #saving the data
    np.savetxt(output, new_grid, delimiter=' ', fmt='%1d', newline=' \n')
#Workers sending their data to master                
else:
    comm.Send(np.copy(data), dest = 0)
