import sqlite3

conn = sqlite3.connect('metadata.db')
c = conn.cursor()
for gid in [326461, 329404, 326459, 305498, 326460]:
    c.execute('SELECT global_id, video_id, frame_id, image_path FROM metadata WHERE global_id = ?', (gid,))
    print(f'GID {gid}:', c.fetchone())

c.execute("SELECT name FROM sqlite_master WHERE type='table'")
print('Tables in metadata.db:', c.fetchall())

# Check if keyframe_id 'L21_V016_transnetv2_0140_00' exists in metadata.db
c.execute("SELECT global_id, video_id, frame_id FROM metadata WHERE frame_id LIKE '%L21_V016_transnetv2_0140_00%'")
print('L21_V016_transnetv2_0140_00 in metadata.db:', c.fetchall())

# Check L27_V015_transnetv2_0124_04 in metadata.db
c.execute("SELECT global_id, video_id, frame_id FROM metadata WHERE frame_id LIKE '%L27_V015_transnetv2_0124_04%'")
print('L27_V015_transnetv2_0124_04 in metadata.db:', c.fetchall())

conn.close()
