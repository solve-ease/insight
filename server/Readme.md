So here we will basically expose the image search tool via an API.
the frontend ideally will be a web app served on the localhost,

it should:
- provide endpoints for searching images and videos
- provide an endpoint to manually rescan and reindex and delete vector db
- setup up automated scanning of files based on system events
- rescan on every startup
- reject all requests from outside of the localhost