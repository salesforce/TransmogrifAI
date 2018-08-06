# TransmogrifAI Docs

- `git clone https://github.com/salesforce/TransmogrifAI.git` - clone TransmogrifAI repo
- `cd ./TransmogrifAI` - go to cloned directory
- `./gradlew docs:buildDocs` - build documentation files
- `./gradlew docs:serve` - run a web server to serve the docs (Ctrl-C to stop).
- `open http://localhost:3000` or visit http://localhost:3000 in your browser

You can also run `./gradlew docs:buildDocs --continuous` in one terminal to automatically rebuild the docs when
something changes, then run `./gradlew docs:serve` in another terminal to run the web server.
