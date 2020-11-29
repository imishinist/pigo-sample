package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"io/ioutil"
	"log"

	pigo "github.com/esimov/pigo/core"

	"gocv.io/x/gocv"
)

var (
	cascadeFile = flag.String("cf", "", "cascade binary file")
	minSize     = flag.Int("min", 20, "Minimum size of face")
	maxSize     = flag.Int("max", 1000, "Maximum size of face")
	shiftFactor = flag.Float64("shift", 0.1, "Shift detection window by percentage")
	scaleFactor = flag.Float64("scale", 1.1, "Scale detection window by percentage")
)

func main() {
	flag.Parse()

	cascadeFile, err := ioutil.ReadFile(*cascadeFile)
	if err != nil {
		log.Fatalf("Error reading the cascade file: %v", err)
	}

	// webcam, err := gocv.VideoCaptureFile(flag.Arg(0))
    webcam, err := gocv.OpenVideoCapture(0)
	if err != nil {
		log.Fatal(err)
	}
    defer webcam.Close()

	window := gocv.NewWindow("Face Detect")
	defer window.Close()

	mat := gocv.NewMat()
	defer mat.Close()

	blue := color.RGBA{0, 0, 255, 0}

	pg := pigo.NewPigo()
	classifier, err := pg.Unpack(cascadeFile)
	if err != nil {
		log.Fatalf("Error reading the cascade file: %s", err)
	}

	for {
		if ok := webcam.Read(&mat); !ok {
			fmt.Println(err)
			return
		}
		if mat.Empty() {
			continue
		}

		img, err := mat.ToImage()
		if err != nil {
			continue
		}
		src := pigo.ImgToNRGBA(img)
		frame := pigo.RgbToGrayscale(src)

		cols, rows := src.Bounds().Max.X, src.Bounds().Max.Y
		cParams := pigo.CascadeParams{
			MinSize:     *minSize,
			MaxSize:     *maxSize,
			ShiftFactor: *shiftFactor,
			ScaleFactor: *scaleFactor,
			ImageParams: pigo.ImageParams{
				Pixels: frame,
				Rows:   rows,
				Cols:   cols,
				Dim:    cols,
			},
		}
		dets := classifier.RunCascade(cParams, 0.0)
		dets = classifier.ClusterDetections(dets, 0.2)

		var qThresh float32 = 5.0

		for i := 0; i < len(dets); i++ {
			if dets[i].Q > qThresh {
				gocv.Rectangle(&mat, image.Rect(
					dets[i].Col-dets[i].Scale/2,
					dets[i].Row-dets[i].Scale/2,
					dets[i].Col+dets[i].Scale/2,
					dets[i].Row+dets[i].Scale/2,
				), blue, 3)
			}
		}

		window.IMShow(mat)
		if window.WaitKey(1) >= 0 {
			break
		}
	}
}
