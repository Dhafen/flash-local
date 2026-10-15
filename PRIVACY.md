# Flash Local privacy

Conversations and inference stay on the device. Flash Local does not send prompts
or replies to an inference service. There are no advertising or analytics SDKs.

Fetching model metadata or downloading weights contacts Hugging Face and its
download hosts. Those requests expose normal network information such as the
device's public IP address. Conversations are not part of those requests.

On iPhone, the default model location is this app's Documents/Models folder,
visible in Files under On My iPhone. Model files are excluded from device backup.
Deleting the app can delete its local data. Diagnostic exports are local files
until you choose to share them. Apple's TestFlight crash reporting follows
your Apple/TestFlight settings.

This development fork is based on Minirun. Optional LAN serving is not enabled
in this build; its access controls and privacy documentation must be reviewed
before that feature ships.
