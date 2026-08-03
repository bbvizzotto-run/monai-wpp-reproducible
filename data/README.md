# Data directory

Clinical images and identifiers must remain outside version control.

Use `metadata.example.csv` as the schema for a local manifest. Store private files under
`data/private/` or another protected location ignored by Git.

The definitive manifest should use anonymous identifiers and preserve the pairing between the
original image and its WhatsApp-compressed counterpart.
