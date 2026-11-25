# Privacy memo

## 1. What it does, in plain language

Our app draws boxes on a copy of an image to show where the software thinks it found faces. The numbers label boxes in that image; they do not identify anyone or follow them across frames. The overlay leaves faces and backgrounds visible. It lets us inspect what the software found or missed, but it does not explain why a box appeared.

The overlay returns the annotated image, box locations, and a count without saving images, recording video, or logging detections. The submitted `artifact.png` is a saved example made from supplied data. The separate `detections.csv` records source filenames, object labels, scores, and box locations for supplied scenes, not webcam activity.

## 2. Where the data came from

According to `provided/PROVENANCE.md`, the test faces were generated, resized, and placed on generated backgrounds. Nobody posed for those test images. But that only answers the consent question at the evaluation stage. The generator learned from real photographs, and the provenance file does not establish whether the people in those photographs agreed to that use. Synthetic test images avoid directly using someone's portrait in our audit; they do not settle how the training photos were obtained or give us permission to film anyone in a new setting.

## 3. What we measured, and what we did not

In our L5 run, the detector returned at least one box on 100% of `near` images (faces 200–260 pixels across) and about 7% of `far` images (90–120 pixels), a gap of about 93 percentage points. A hit means the detector returned a box somewhere in the image, not that the box landed on the face. The groups differ in apparent face size, not measured camera distance. These rates describe this synthetic test set, not what we should expect at a particular distance in a real room.

Yu et al. (2022), [“Enhancing Fairness in Face Detection in Computer Vision Systems by Demographic Bias Mitigation”](https://doi.org/10.1145/3514094.3534153), reported performance disparities across demographic groups in face detection. Our audit neither confirms nor rules out those disparities in our detector. We did not split the synthetic images by apparent skin tone because any gap could reflect how the generator renders faces, how the detector handles those renderings, or both.

## 4. What we would want in place before this ran on real people

Only a designated operator could start a private session, after everyone in frame opts in. Only the operator and teammates actively debugging could see the overlay. If a box is wrong, the team can keep observing to diagnose the failure, but must not use the output to make decisions about people. If anyone in frame has not consented or withdraws consent, the operator stops the feed and any recording.

Recording would be off by default. With separate consent, the operator could save the original input and overlay for debugging. Both files would delete automatically after 24 hours unless the operator approves a one-time extension to seven days from recording and everyone visible agrees. Before the session, participants would receive the operator's contact information for complaints or early deletion requests. These are proposed safeguards, not features of the submitted app.
