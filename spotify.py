import csv
import numpy as np
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import time
from sklearn.model_selection import train_test_split
from sklearn import tree


# spotify developerから取得したclient_idとclient_secretを入力
client_id = 'XXXXXXXXXXXXXXXX'
client_secret = 'XXXXXXXXXXXXXXXXXXXXX'

client_credentials_manager = SpotifyClientCredentials(client_id, client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def getTrackIDs(playlist_ids):
  track_ids = []
  
  for playlist_id in playlist_ids:
    playlist = sp.playlist(playlist_id)
    while playlist['tracks']['next']:
      for item in playlist['tracks']['items']:
        track = item['track']
        if not track['id'] in track_ids:
          track_ids.append(track['id'])
      playlist['tracks'] = sp.next(playlist['tracks'])
    else:
      for item in playlist['tracks']['items']:
        track = item['track']
        if not track['id'] in track_ids:
          track_ids.append(track['id'])

  return track_ids

# SpotifyのプレイリストのIDを入力 プレイリストの曲の特徴量をとって
# データセットとしてcsvファイルに書き出す。
# ここでは、2019年と2020年のトップソングと、自分で作成した好きじゃない曲プレイリストをplaylist_idsに入れた。
# playlist_ids = [ 'XXXXXXXXXXXXXXXXX',]  
playlist_ids = [ 'XXXXXXXXXXXXXXXX','XXXXXXXXXXXXXXXXXX']  # プレイリスト2つを書き出す場合

track_ids = getTrackIDs(playlist_ids)

def getTrackFeatures(id):
  meta = sp.track(id)
  features = sp.audio_features(id)

  name = meta['name']
  album = meta['album']['name']
  artist = meta['album']['artists'][0]['name']
  release_date = meta['album']['release_date']
  length = meta['duration_ms']
  popularity = meta['popularity']
  key = features[0]['key']
  mode = features[0]['mode']
  danceability = features[0]['danceability']
  acousticness = features[0]['acousticness']
  energy = features[0]['energy']
  instrumentalness = features[0]['instrumentalness']
  liveness = features[0]['liveness']
  loudness = features[0]['loudness']
  speechiness = features[0]['speechiness']
  tempo = features[0]['tempo']
  time_signature = features[0]['time_signature']
  valence = features[0]['valence']

  track = [name, album, artist, release_date, length, popularity, key, mode, danceability, acousticness, energy, instrumentalness, liveness, loudness, speechiness, tempo, time_signature, valence]
  return track

tracks = []

for track_id in track_ids:
  time.sleep(1)
  track = getTrackFeatures(track_id)
  tracks.append(track)

df1 = pd.DataFrame(tracks, columns = ['name', 'album', 'artist', 'release_date', 'length', 'popularity', 'key', 'mode', 'danceability', 'acousticness', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'time_signature', 'valence'])
df1.to_csv('/Users/topsong.csv')

df2 = pd.DataFrame(tracks, columns = ['name', 'album', 'artist', 'release_date', 'length', 'popularity', 'key', 'mode', 'danceability', 'acousticness', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'time_signature', 'valence'])
df2.to_csv('/Users/topsong.csv')

# プレイリストに"like"カラムの追加。好きな曲は＂1＂、好きではない曲は"0"
df2["like"]=0
df1["like"]=1

#プレイリストを１つのデータフレームにまとめる
df3 = pd.concat([df1, df2])
#要らない特徴量を削除
df4 = df3.drop(['name', 'album','artist','release_date'], axis=1)

#訓練データと検証データを7:3の割合で分割
like_train,  like_test= train_test_split(df4, test_size=0.3)
#訓練データを答えなしのX_trainと答えのy_trainにする
#Xが大文字なのは行列、yが小文字なのはベクトルに由来
X_train = like_train.drop(columns=["like"])
y_train = like_train.like

#実際のテストデータの答えをlike、テストデータの答えなしをlike_testに代入
like = like_test["like"]
like_test = like_test.drop(columns=["like"])

#モデルの作成と学習
model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)

#予測値をy_predに代入
y_pred = model.predict(like_test)

#予測値とテストデータの結果数を確認
len(like_test), len(y_pred)

#データセットに予測値の結果（pred）と答え（answer)のカラムを追加
like_test["pred"] = y_pred
like_test["answer"] = like

like_test.to_csv('/Users/answer.csv')