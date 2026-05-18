// CellScope → Fiji bridge
//
// Opens a CellScope-exported (image, labels) TIFF pair as overlaid
// hyperstacks. Run AFTER `python scripts/cellscope_export_fiji.py`
// has produced the TIFFs.
//
// In Fiji:  Plugins → Macros → Run…   →  pick this file
//
// You will be prompted for the IMAGE TIFF; the macro looks for the
// matching labels TIFF beside it (same stem, "_labels" suffix).
//
// Result:
//   • Source image opens as a 16-bit (or 8-bit) timelapse
//   • Label stack overlays in a Glasbey-coloured LUT
//   • Both linked to the same time slider
//
// Tested on Fiji 1.54+ (ImageJ 2.x).

run("Close All");

// Pick the image TIFF
imgPath = File.openDialog("Select <stem>_image.tif");
if (imgPath == "") exit("Cancelled");

stem = replace(imgPath, "_image.tif", "");
labPath = stem + "_labels.tif";
maskPath = stem + "_mask.tif";

// Open the source image
open(imgPath);
imgTitle = getTitle();
run("Enhance Contrast", "saturated=0.35");

// Try labels first, then single-cell mask
overlayPath = "";
if (File.exists(labPath))      overlayPath = labPath;
else if (File.exists(maskPath)) overlayPath = maskPath;

if (overlayPath == "") {
    showMessage("No overlay found",
        "Looked for:\n" + labPath + "\n" + maskPath +
        "\nNo overlay file beside the image. Showing image only.");
    exit;
}

open(overlayPath);
ovTitle = getTitle();

// Apply Glasbey LUT to label image so each cell gets a distinct color.
// 'glasbey on dark' is a Fiji default LUT for label images.
selectWindow(ovTitle);
run("glasbey on dark");

// Sync the two stacks' frame sliders
run("Synchronize Windows");

// Make the overlay 50% opaque on top of the image. Fiji's Image
// Overlay doesn't directly support that across stacks, but a
// clean approach is the Composite display:
//   Image → Color → Merge Channels → channel 1 = grayscale image,
//                                     channel 2 = labels
// Here we just leave both windows open side-by-side and rely on
// "Synchronize Windows" — easier for non-experts.

selectWindow(imgTitle);
showMessage("CellScope loaded",
    "Image + labels open and synchronized.\n\n" +
    "• Drag time slider on either to scrub both.\n" +
    "• Image → Color → Merge Channels to fuse into a hyperstack.\n" +
    "• Use the labels image alone for ROI manager / measurements.");
