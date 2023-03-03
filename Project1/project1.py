import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

#Load data from csv file
df = pd.read_csv('/Users/pacodiaz/Downloads/music_genre.csv', usecols=['danceability', 'energy', 'instrumentalness','popularity','music_genre'])

#remove rows with null values
df = df.dropna()

#remove rows with 0 popularity
df = df[df['popularity'] != 0]

#remove rows with 0 danceability
df = df[df['danceability'] != 0]

#remove rows with 0 energy
df = df[df['energy'] != 0]

#remove rows with 0 instrumentalness
df = df[df['instrumentalness'] != 0]

#remove rows with 0 genre
df = df[df['music_genre'] != '']


features = ['danceability', 'energy', 'instrumentalness','music_genre']

#my data
df_x = df[features]


#convert genre alternative to 0 in df
df_x.loc[df_x['music_genre'] == 'Alternative', 'music_genre'] = 0

#convert genre anime to 1 in df
df_x.loc[df_x['music_genre'] == 'Anime', 'music_genre'] = 1

#convert genre blues to 2 in df
df_x.loc[df_x['music_genre'] == 'Blues', 'music_genre'] = 2

#convert genre classical to 3 in df
df_x.loc[df_x['music_genre'] == 'Classical', 'music_genre'] = 3

#convert genre country to 4 in df
df_x.loc[df_x['music_genre'] == 'Country', 'music_genre'] = 4

#convert genre electronic to 5 in df
df_x.loc[df_x['music_genre'] == 'Electronic', 'music_genre'] = 5

#convert genre hip-hop to 6 in df
df_x.loc[df_x['music_genre'] == 'Hip-Hop', 'music_genre'] = 6

#convert genre jazz to 7 in df
df_x.loc[df_x['music_genre'] == 'Jazz', 'music_genre'] = 7

#convert genre rap to 8 in df
df_x.loc[df_x['music_genre'] == 'Rap', 'music_genre'] = 8

#convert genre rock to 9 in df
df_x.loc[df_x['music_genre'] == 'Rock', 'music_genre'] = 9


df_y = df['popularity']

#Use 80% of data for training
test_size_80 = math.floor(len(df) * 0.8)


# Split the data into training/testing sets
x_train = df_x[:test_size_80]
x_test = df_x[test_size_80:]
y_train = df_y[:test_size_80]
y_test = df_y[test_size_80:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model
regr.fit(x_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(x_test)



#get R2
print('R2: ', round(regr.score(x_test, y_test),2))

answer = True

while answer == True:
    #user input for features
    danceability = float(input("Enter danceability: "))
    energy = float(input("Enter energy: "))
    instrumentalness = float(input("Enter instrumentalness: "))
    



    #user input for genre
    genre_input = input("Enter genre: ")

    #convert genre to number
    if genre_input.__eq__('Alternative') or genre_input.__eq__('alternative'):
        genre = 0
    
    elif genre_input.__eq__('Anime') or genre_input.__eq__('anime'):
        genre = 1
    
    elif genre_input.__eq__('Blues') or genre_input.__eq__('blues'):
        genre = 2
    
    elif genre_input.__eq__('Classical') or genre_input.__eq__('classical'):
        genre = 3

    elif genre_input.__eq__('Country') or genre_input.__eq__('country'):
        genre = 4

    elif genre_input.__eq__('Electronic') or genre_input.__eq__('electronic'):
        genre = 5
    
    elif genre_input.__eq__('Hip-Hop') or genre_input.__eq__('hip-hop'):
        genre = 6

    elif genre_input.__eq__('Jazz') or genre_input.__eq__('jazz'):
        genre = 7

    elif genre_input.__eq__('Rap') or genre_input.__eq__('rap'):
        genre = 8
    
    elif genre_input.__eq__('Rock') or genre_input.__eq__('rock'):
        genre = 9

    else:
        print("Invalid genre")
        break

    #create dataframe with user input
    df_pred = pd.DataFrame([[danceability, energy, instrumentalness,genre]], columns=features)

    #predict popularity using user input


    
    popularity = regr.predict(df_pred)[0]

    #print popularity with only 2 deciamals
    print("Predicted popularity: ", round(popularity, 2))

    #plot predicted popularity vs actual popularity
    plt.scatter(y_test, y_pred, color='blue')
    plt.title('Predicted Popularity vs Actual Popularity')
    plt.xlabel('Actual Popularity')
    plt.ylabel('Predicted Popularity')
    plt.show()


   


    #plot music genre vs popularity
    plt.scatter(df['music_genre'], df['popularity'], color='black')
    plt.title('Music Genre vs Popularity')
    plt.xlabel('Music Genre')
    plt.ylabel('Popularity')
    plt.show()

    #separate data by genre
    df_alt = df[df['music_genre'] == 'Alternative']
    df_ani = df[df['music_genre'] == 'Anime']
    df_blu = df[df['music_genre'] == 'Blues']
    df_cla = df[df['music_genre'] == 'Classical']
    df_cou = df[df['music_genre'] == 'Country']
    df_ele = df[df['music_genre'] == 'Electronic']
    df_hip = df[df['music_genre'] == 'Hip-Hop']
    df_jaz = df[df['music_genre'] == 'Jazz']
    df_rap = df[df['music_genre'] == 'Rap']
    df_rock = df[df['music_genre'] == 'Rock']

    #plot prediction for genre vs actual popularity

    if genre == 0:
        #plot df_alt
        plt.scatter(df_alt['danceability'], df_alt['popularity'], color='black')
        plt.title('Alternative Music')
        plt.xlabel('Danceability')
        plt.ylabel('Popularity')
        plt.show()

        #plot df_alt for energy vs popularity
        plt.scatter(df_alt['energy'], df_alt['popularity'], color='black')
        plt.title('Alternative Music')
        plt.xlabel('Energy')
        plt.ylabel('Popularity')
        plt.show()

        #plot df_alt for instrumentalness vs popularity
        plt.scatter(df_alt['instrumentalness'], df_alt['popularity'], color='black')
        plt.title('Alternative Music')
        plt.xlabel('Instrumentalness')
        plt.ylabel('Popularity')
        plt.show()



    #here I'm plotting my features vs popularity for each genre, these are not predictions
    #this is the actual data for each genre        
    elif genre == 1:
        #plot df_ani
        plt.scatter(df_ani['danceability'], df_ani['popularity'], color='black')
        plt.title('Anime Music')
        plt.xlabel('Danceability')
        plt.ylabel('Popularity')
        plt.show()

        #plot df_ani for energy vs popularity
        plt.scatter(df_ani['energy'], df_ani['popularity'], color='black')
        plt.title('Anime Music')
        plt.xlabel('Energy')
        plt.ylabel('Popularity')
        plt.show()

        #plot df_ani for instrumentalness vs popularity
        plt.scatter(df_ani['instrumentalness'], df_ani['popularity'], color='black')
        plt.title('Anime Music')
        plt.xlabel('Instrumentalness')
        plt.ylabel('Popularity')
        plt.show()

    elif genre == 2:
        #plot df_blu
        plt.scatter(df_blu['danceability'], df_blu['popularity'], color='black')
        plt.title('Blues Music')
        plt.xlabel('Danceability')
        plt.ylabel('Popularity')
        plt.show()

        #plot df_blu for energy vs popularity
        plt.scatter(df_blu['energy'], df_blu['popularity'], color='black')
        plt.title('Blues Music')
        plt.xlabel('Energy')
        plt.ylabel('Popularity')
        plt.show()

        #plot df_blu for instrumentalness vs popularity
        plt.scatter(df_blu['instrumentalness'], df_blu['popularity'], color='black')
        plt.title('Blues Music')
        plt.xlabel('Instrumentalness')
        plt.ylabel('Popularity')
        plt.show()

    elif genre == 3:
        #plot df_cla
        plt.scatter(df_cla['danceability'], df_cla['popularity'], color='black')
        plt.title('Classical Music')
        plt.xlabel('Danceability')
        plt.ylabel('Popularity')
        plt.show()

        #plot df_cla for energy vs popularity
        plt.scatter(df_cla['energy'], df_cla['popularity'], color='black')
        plt.title('Classical Music')
        plt.xlabel('Energy')
        plt.ylabel('Popularity')
        plt.show()

        #plot df_cla for instrumentalness vs popularity
        plt.scatter(df_cla['instrumentalness'], df_cla['popularity'], color='black')
        plt.title('Classical Music')
        plt.xlabel('Instrumentalness')
        plt.ylabel('Popularity')
        plt.show()

    elif genre == 4:
        #plot df_cou
        plt.scatter(df_cou['danceability'], df_cou['popularity'], color='black')
        plt.title('Country Music')
        plt.xlabel('Danceability')
        plt.ylabel('Popularity')
        plt.show()

        #plot df_cou for energy vs popularity
        plt.scatter(df_cou['energy'], df_cou['popularity'], color='black')
        plt.title('Country Music')
        plt.xlabel('Energy')
        plt.ylabel('Popularity')
        plt.show()

        #plot df_cou for instrumentalness vs popularity
        plt.scatter(df_cou['instrumentalness'], df_cou['popularity'], color='black')
        plt.title('Country Music')
        plt.xlabel('Instrumentalness')
        plt.ylabel('Popularity')
        plt.show()
    
    elif genre == 5:
        #plot df_ele
        plt.scatter(df_ele['danceability'], df_ele['popularity'], color='black')
        plt.title('Electronic Music')
        plt.xlabel('Danceability')
        plt.ylabel('Popularity')
        plt.show()

        #plot df_ele for energy vs popularity
        plt.scatter(df_ele['energy'], df_ele['popularity'], color='black')
        plt.title('Electronic Music')
        plt.xlabel('Energy')
        plt.ylabel('Popularity')
        plt.show()

        #plot df_ele for instrumentalness vs popularity
        plt.scatter(df_ele['instrumentalness'], df_ele['popularity'], color='black')
        plt.title('Electronic Music')
        plt.xlabel('Instrumentalness')
        plt.ylabel('Popularity')
        plt.show()
    
    elif genre == 6:
        #plot df_hip
        plt.scatter(df_hip['danceability'], df_hip['popularity'], color='black')
        plt.title('Hip-Hop Music')
        plt.xlabel('Danceability')
        plt.ylabel('Popularity')
        plt.show()

        #plot df_hip for energy vs popularity
        plt.scatter(df_hip['energy'], df_hip['popularity'], color='black')
        plt.title('Hip-Hop Music')
        plt.xlabel('Energy')
        plt.ylabel('Popularity')
        plt.show()

        #plot df_hip for instrumentalness vs popularity
        plt.scatter(df_hip['instrumentalness'], df_hip['popularity'], color='black')
        plt.title('Hip-Hop Music')
        plt.xlabel('Instrumentalness')
        plt.ylabel('Popularity')
        plt.show()
    
    elif genre == 7:
        #plot df_jaz
        plt.scatter(df_jaz['danceability'], df_jaz['popularity'], color='black')
        plt.title('Jazz Music')
        plt.xlabel('Danceability')
        plt.ylabel('Popularity')
        plt.show()

        
    
    elif genre == 8:
        #plot df_rap
        plt.scatter(df_rap['danceability'], df_rap['popularity'], color='black')
        plt.title('Rap Music')
        plt.xlabel('Danceability')
        plt.ylabel('Popularity')
        plt.show()

        #plot df_rap for energy vs popularity
        plt.scatter(df_rap['energy'], df_rap['popularity'], color='black')
        plt.title('Rap Music')
        plt.xlabel('Energy')
        plt.ylabel('Popularity')
        plt.show()

        #plot df_rap for instrumentalness vs popularity
        plt.scatter(df_rap['instrumentalness'], df_rap['popularity'], color='black')
        plt.title('Rap Music')
        plt.xlabel('Instrumentalness')
        plt.ylabel('Popularity')
        plt.show()

    elif genre == 9:
        #plot df_rock
        plt.scatter(df_rock['danceability'], df_rock['popularity'], color='black')
        plt.title('Rock Music')
        plt.xlabel('Danceability')
        plt.ylabel('Popularity')
        plt.show()

        #plot df_rock for energy vs popularity
        plt.scatter(df_rock['energy'], df_rock['popularity'], color='black')
        plt.title('Rock Music')
        plt.xlabel('Energy')
        plt.ylabel('Popularity')
        plt.show()

        #plot df_rock for instrumentalness vs popularity
        plt.scatter(df_rock['instrumentalness'], df_rock['popularity'], color='black')
        plt.title('Rock Music')
        plt.xlabel('Instrumentalness')
        plt.ylabel('Popularity')
        plt.show()

    #plot prediction 



    print("\nMake another prediction? (y/n)\n")
    answer = input()
    if answer == 'y' or answer == 'Y' or answer == 'yes' or answer == 'Yes':
        answer = True
    else:
        answer = False













