# External model license

The model is downloaded separately; its license differs from the app's
Apache-2.0 source license. The complete Qwen Community License 1.0 is bundled
as `Apps/Minirun/Resources/QwenModelLicense.txt` and available offline from the
iPhone chat screen's information menu, alongside the app's source notices.

The unmodified 3,235-byte UTF-8 file comes from the original base model:

https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/de4b8e4d43b917e7706784d8bb445c9af86a3540/LICENSE

SHA-256: `a0dc422560841fd68e06d974907f8b4c709bca44a67daad2b528437bdf676c08`.
The bundle test checks these exact bytes. This records the base model's notice;
it does not relicense the REAM derivative or declare commercial rights beyond
the actual license. The downloaded artifact remains pinned to its separate
manifest and SHA-256, with its published model name visible in the chat screen.

`GGUFModelManifest.primaryREAM` now carries the artifact repository/revision,
the inspected REAM source and base repository/revision references, plus
this notice's name, pinned source URL and SHA-256. Measurement and numerical
reports copy that declared provenance from the verified manifest. It is an
optional additive field: old reports remain readable, and synthetic fixtures
without a declaration do not inherit the primary model's identity. Metadata
does not attest the publisher's exact upstream conversion lineage, authorize
file access, replace full SHA verification, prove inference
or change the artifact's license. Tests bind the declared artifact to its
actual download URL and the license digest to the actual offline bundle bytes.
