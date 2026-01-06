So here we will basically expose the image search tool via an API.
the frontend ideally will be a web app served on the localhost,

OK so the idea is to create a docker compose system where the user can start the service and interact with it using a local website which is connected to the backedn running in another container now this backend will scan for media index them and perform all the core funcitons.

the server should:
- provide endpoints for searching images and videos
- provide an endpoint to manually rescan and reindex and delete vector db
- setup up automated scanning of files based on system events (use inotify to trigger events when there is some change in the file system.) # This feature will not be a part of this version -> it will be complex and may introduce performance challenges.
- rescan on every startup
- reject all requests from outside of the localhost
- should be able to set a scope so that i works only for files within that scope and not all others, thus reducing the vector db size and processing.
- the server should maintain its state across restarts and not loose the vecotr db on every restart.


Components:
- file database (postgres)
- Vector database (qdrant)
- delta_prefilter (superficially scan files for changes)
- delta_monitor (identify if a file has changed by comparing hash from the file database)
- embedding_model (converts a image file into an embedding)
- video_embedding (works as a wrapper for embedding model, this carries out the necessary sampling and clustering for the embedding images)
- vector_db_Service (I/O operations in vecotr database)
- text_search (search from the vector database using text)

functionality:
- Scan and index files: Here we have to scan all the media files present in the given folder and then index them in the vector database.
- Search: search from the files in the vector database using a text querry.

How it works:
- Scanning files:
    - here the objective is to detect all the new or changed or moved or renamed files.
    1. we use the prefilter to cheapy detect if a file has been changed or not. we mointor the mtime (last modified time) and the size  and the path of all the files we come across.
        - Here path will be the key for the search operation.
        - we will sample all files one by one and then look them up in the files db,
        - if present we will compare the mtime and size to detect any changes. if there is a change the files has been changed and should be sent to the next stage. The latest values will be updated in the files database.
        - now we should update the scantime attribute of the file tuple to the time it was rescanned.
        - if the file is not present in the files database we will create a new tuple for it and get it to the next stage.
        - after all this process is done we will retreive all the entries with scantime earlier than this scan was started at. This would give a list of all the files entries which were not accessed, this means that file has been deleted or moved or renamed. So here we will use the point id attribute of the tuple to delete the corresponding point from the vector database and also delete all the entries from the files database. 

    2. After the prefilter is done running we have all the files that are either new or changed with us in the next stage.
        - now we will compute the hash of each files in this stage.
        - this hash is compared with the hash stored in the files database if there is a change then the files has been changed and it will be indexed again. and the old embedding will be deleted using the point id attribute of the old tuple. the new hash and point id should be updated
        - if a hash is not found then the file is new and then we would have to get it indexed.
        - if the hash is same then the files has only been renamed or moved so dont do anything.

roadmap:
1. first create all the 