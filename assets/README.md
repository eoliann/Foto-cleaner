# Modele AI incluse

Toate modelele sunt incluse în executabil și rulează local prin runtime-ul `RTen`.

## MODNet: eliminarea fundalului

- Sursă: [Xenova/modnet](https://huggingface.co/Xenova/modnet)
- Revizie: `fa2fa546052fba4c08921230a26cc69a333fca12`
- Fișier: `onnx/model.onnx` (`modnet.onnx`)
- SHA-256: `07c308cf0fc7e6e8b2065a12ed7fc07e1de8febb7dc7839d7b7f15dd66584df9`
- Licență declarată: Apache-2.0 ([`MODNET-LICENSE.txt`](MODNET-LICENSE.txt))
- Proiect original: [ZHKKKe/MODNet](https://github.com/ZHKKKe/MODNet)

## YuNet: detectarea feței

- Sursă: [opencv/opencv_zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- Fișier: `face_detection_yunet_2023mar.onnx` (`yunet.onnx`), intrare fixă 640x640
- SHA-256: `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`
- Licență: MIT ([`YUNET-LICENSE.txt`](YUNET-LICENSE.txt))
- Proiect original: [ShiqiYu/libfacedetection.train](https://github.com/ShiqiYu/libfacedetection.train)

## MI-GAN: eliminarea watermark-urilor și a obiectelor

- Proiect original: [Picsart-AI-Research/MI-GAN](https://github.com/Picsart-AI-Research/MI-GAN) (ICCV 2023), modelul MI-GAN-512 antrenat pe Places2
- Sursa greutăților: `migan_traced.pt` din [Sanster/models, release `migan`](https://github.com/Sanster/models/releases/tag/migan) (MD5 `76eb3b1a71c400ee3290524f7a11b89c`), folosit și de IOPaint
- Conversie: `torch.onnx.export` (opset 17), intrare `[1, 4, 512, 512]` = (cunoscut − 0.5, imagine × cunoscut), imaginea în [-1, 1]; ieșire `[1, 3, 512, 512]` în [-1, 1]. Ieșirea în RTen coincide cu PyTorch (diferență maximă ~2e-5).
- Fișier: `migan.onnx`
- SHA-256: `8cf778c94b2b20d93ee7dca47d85468e71f2d6a07f0f91323d3b886d1ef69ab1`
- Licență: MIT, atât pentru cod cât și pentru greutăți ([`MIGAN-LICENSE.txt`](MIGAN-LICENSE.txt))
