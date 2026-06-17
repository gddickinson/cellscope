# IC293 cropped single cells — staging provenance

Source (read-only): `/Volumes/pathaklab/Lab/Ignasi/IC293_ECmigration/IC293_CroppedSelectedCells`  
Staged to: `ic293_analysis/_cache/`  
Staged: 2026-06-16T14:02:16  •  git `2cdbd315fe`  
tifffile 2026.3.3, numpy 2.4.6

## What this is

Ignasi's hand-cropped single cells from the **IC293_ECmigration** dataset, converted to the IC295 on-disk format so the existing CellScope detection/analysis pipeline runs on them unchanged.

**Single-channel DIC** — the IC293 originals were 2-channel `['DIC 10x','None']` and only the DIC channel was cropped. There is no Cy5/fluorescence channel (no SirActin in IC293), so the Cy5-dependent pipeline steps (alignment, persistence-guard) auto-skip; detection runs on DIC as usual.

Same microscope as IC295: **0.6523 µm/px**, **10 min** frame interval.

## Per-crop files (in `_cache/`)

- `<label>.ome.tif` — clean OME-TIFF `(T,H,W)` uint16, axes=TYX, physical pixel size + frame interval embedded.
- `<label>.ome.json` — sidecar: `n_channels=1`, `dic_channel=0`, `fluo_channel=null`, scale/interval, source provenance.
- `<label>_metadata.txt` — MM-style Summary with `Channels=1` (so `core.io.detect_channels` resolves single-channel).
- `_source_metadata/<orig>_metadata.txt` — untouched original MM metadata, kept for provenance.

## Label scheme

`Pos{N}-{COND}` (primary crop), `Pos{N}_cell{K}-{COND}` (K-th cell), `Pos{N}_div-{COND}` (dividing crop). Condition is always the final `-` token, so `parse_condition` works.

## Inventory

### WT (18)

| label | source file | T×H×W |
|---|---|---|
| `Pos0-WT` | `IC293__1_MMStack_Pos0-WT.ome-cropped.tif` | 74×438×759 |
| `Pos3-WT` | `IC293__1_MMStack_Pos3-WT.ome-cropped.tif` | 60×668×822 |
| `Pos4-WT` | `IC293__1_MMStack_Pos4-WT.ome-cropped.tif` | 42×702×714 |
| `Pos5-WT` | `IC293__1_MMStack_Pos5-WT.ome-cropped.tif` | 97×970×374 |
| `Pos6_cell1-WT` | `IC293__1_MMStack_Pos6-WT.ome-cell1-cropped.tif` | 36×358×297 |
| `Pos6_cell2-WT` | `IC293__1_MMStack_Pos6-WT.ome-cell2-cropped.tif` | 70×331×279 |
| `Pos7-WT` | `IC293__1_MMStack_Pos7-WT.ome-cropped.tif` | 70×770×816 |
| `Pos8_cell1-WT` | `IC293__1_MMStack_Pos8-WT.ome-cell1-cropped.tif` | 58×508×664 |
| `Pos8_cell2-WT` | `IC293__1_MMStack_Pos8-WT.ome-cell2-cropped.tif` | 30×582×620 |
| `Pos8_cell3-WT` | `IC293__1_MMStack_Pos8-WT.ome-cell3-cropped.tif` | 42×582×476 |
| `Pos9_cell1-WT` | `IC293__1_MMStack_Pos9-WT.ome-cell1-cropped.tif` | 49×272×520 |
| `Pos9_cell2-WT` | `IC293__1_MMStack_Pos9-WT.ome-cell2-cropped.tif` | 57×494×512 |
| `Pos10-WT` | `IC293__1_MMStack_Pos10-WT.ome-cropped.tif` | 97×728×836 |
| `Pos11_cell1-WT` | `IC293__1_MMStack_Pos11-WT.ome-cell1-cropped.tif` | 97×594×690 |
| `Pos11_cell2-WT` | `IC293__1_MMStack_Pos11-WT.ome-cell2-cropped.tif` | 44×604×492 |
| `Pos12-WT` | `IC293__1_MMStack_Pos12-WT.ome-cropped.tif` | 55×672×844 |
| `Pos13_cell1-WT` | `IC293__1_MMStack_Pos13-WT.ome-cell1-cropped.tif` | 60×730×450 |
| `Pos13_cell2-WT` | `IC293__1_MMStack_Pos13-WT.ome-cell2-cropped.tif` | 60×504×456 |

### KO (16)

| label | source file | T×H×W |
|---|---|---|
| `Pos14-KO` | `IC293__1_MMStack_Pos14-KO.ome-cropped.tif` | 57×400×538 |
| `Pos15-KO` | `IC293__1_MMStack_Pos15-KO.ome-cropped.tif` | 55×432×620 |
| `Pos17-KO` | `IC293__1_MMStack_Pos17-KO.ome-cropped.tif` | 65×626×758 |
| `Pos18_cell2-KO` | `IC293__1_MMStack_Pos18-KO.ome-cell2-cropped.tif` | 55×496×380 |
| `Pos18-KO` | `IC293__1_MMStack_Pos18-KO.ome-cropped.tif` | 40×488×644 |
| `Pos19-KO` | `IC293__1_MMStack_Pos19-KO.ome-cropped.tif` | 97×390×652 |
| `Pos20_cell2-KO` | `IC293__1_MMStack_Pos20-KO.ome-cell2-croped.tif` | 97×732×522 |
| `Pos20_cell3-KO` | `IC293__1_MMStack_Pos20-KO.ome-cell3-cropped.tif` | 97×482×526 |
| `Pos20-KO` | `IC293__1_MMStack_Pos20-KO.ome-cropped.tif` | 56×350×606 |
| `Pos21-KO` | `IC293__1_MMStack_Pos21-KO.ome-cropped.tif` | 97×642×402 |
| `Pos22_cell2-KO` | `IC293__1_MMStack_Pos22-KO.ome-cell2-cropped.tif` | 73×550×500 |
| `Pos22-KO` | `IC293__1_MMStack_Pos22-KO.ome-cropped.tif` | 35×598×628 |
| `Pos23-KO` | `IC293__1_MMStack_Pos23-KO.ome-cropped.tif` | 97×552×430 |
| `Pos24_cell2-KO` | `IC293__1_MMStack_Pos24-KO.ome-cell2-cropped.tif` | 64×518×480 |
| `Pos24_cell3-KO` | `IC293__1_MMStack_Pos24-KO.ome-cell3-cropped.tif` | 97×408×462 |
| `Pos24-KO` | `IC293__1_MMStack_Pos24-KO.ome-cropped.tif` | 50×570×608 |

### GOF (11)

| label | source file | T×H×W |
|---|---|---|
| `Pos26_cell2-GOF` | `IC293__1_MMStack_Pos26-GOF.ome-cell2-cropped.tif` | 45×500×510 |
| `Pos26-GOF` | `IC293__1_MMStack_Pos26-GOF.ome-cropped.tif` | 70×532×718 |
| `Pos28-GOF` | `IC293__1_MMStack_Pos28-GOF.ome-cropped.tif` | 30×440×346 |
| `Pos29_cell2-GOF` | `IC293__1_MMStack_Pos29-GOF.ome-cell2-cropped.tif` | 36×516×746 |
| `Pos29-GOF` | `IC293__1_MMStack_Pos29-GOF.ome-cropped.tif` | 38×516×746 |
| `Pos31-GOF` | `IC293__1_MMStack_Pos31-GOF.ome-cropped.tif` | 97×256×332 |
| `Pos33_cell2-GOF` | `IC293__1_MMStack_Pos33-GOF.ome-cell2-cropped.tif` | 53×560×464 |
| `Pos33-GOF` | `IC293__1_MMStack_Pos33-GOF.ome-croped.tif` | 97×592×468 |
| `Pos35_cell1-GOF` | `IC293__1_MMStack_Pos35-GOF.ome-cell1-cropped.tif` | 33×586×868 |
| `Pos35_cell2-GOF` | `IC293__1_MMStack_Pos35-GOF.ome-cell2-cropped.tif` | 38×492×672 |
| `Pos36_div-GOF` | `IC293__1_MMStack_Pos36-GOF.ome-cropped-dividing.tif` | 97×828×980 |

### Y1 (10)

| label | source file | T×H×W |
|---|---|---|
| `Pos49-Y1` | `IC293__1_MMStack_Pos49-Y1.ome-cropped.tif` | 79×526×616 |
| `Pos50-Y1` | `IC293__1_MMStack_Pos50-Y1.ome-cropped.tif` | 50×502×602 |
| `Pos54-Y1` | `IC293__1_MMStack_Pos54-Y1.ome-cropped.tif` | 97×454×664 |
| `Pos55_cell2-Y1` | `IC293__1_MMStack_Pos55-Y1.ome-cell2-cropped.tif` | 55×484×522 |
| `Pos55-Y1` | `IC293__1_MMStack_Pos55-Y1.ome-cropped.tif` | 55×492×522 |
| `Pos56-Y1` | `IC293__1_MMStack_Pos56-Y1.ome-cropped.tif` | 45×448×516 |
| `Pos57-Y1` | `IC293__1_MMStack_Pos57-Y1.ome-cropped.tif` | 52×960×714 |
| `Pos58_cell2-Y1` | `IC293__1_MMStack_Pos58-Y1.ome-cell2-cropped.tif` | 64×390×642 |
| `Pos58-Y1` | `IC293__1_MMStack_Pos58-Y1.ome-cropped.tif` | 40×382×540 |
| `Pos59-Y1` | `IC293__1_MMStack_Pos59-Y1.ome-cropped.tif` | 52×520×560 |

### OT (12)

| label | source file | T×H×W |
|---|---|---|
| `Pos38_cell2-OT` | `IC293__1_MMStack_Pos38-OT.ome-cell2-cropped.tif` | 60×300×348 |
| `Pos38_cell3-OT` | `IC293__1_MMStack_Pos38-OT.ome-cell3-cropped.tif` | 64×440×356 |
| `Pos38-OT` | `IC293__1_MMStack_Pos38-OT.ome-cropped.tif` | 97×550×432 |
| `Pos39-OT` | `IC293__1_MMStack_Pos39-OT.ome-cropped.tif` | 77×414×392 |
| `Pos42-OT` | `IC293__1_MMStack_Pos42-OT.ome-cropped.tif` | 89×432×600 |
| `Pos43_cell2-OT` | `IC293__1_MMStack_Pos43-OT.ome-cell2-cropped.tif` | 65×338×500 |
| `Pos43-OT` | `IC293__1_MMStack_Pos43-OT.ome-cropped.tif` | 97×304×430 |
| `Pos44-OT` | `IC293__1_MMStack_Pos44-OT.ome-cropped.tif` | 97×602×646 |
| `Pos45-OT` | `IC293__1_MMStack_Pos45-OT.ome-cropped.tif` | 73×474×474 |
| `Pos46-OT` | `IC293__1_MMStack_Pos46-OT.ome-cropped.tif` | 97×550×722 |
| `Pos47-OT` | `IC293__1_MMStack_Pos47-OT.ome-cropped.tif` | 35×502×502 |
| `Pos48-OT` | `IC293__1_MMStack_Pos48-OT.ome-cropped.tif` | 97×538×554 |

### DMSO (11)

| label | source file | T×H×W |
|---|---|---|
| `Pos60-DMSO` | `IC293__1_MMStack_Pos60-DMSO.ome-cropped.tif` | 75×552×562 |
| `Pos61-DMSO` | `IC293__1_MMStack_Pos61-DMSO.ome-cropped.tif` | 43×598×546 |
| `Pos62_cell2-DMSO` | `IC293__1_MMStack_Pos62-DMSO.ome-cell2-cropped.tif` | 37×572×514 |
| `Pos62_cell3-DMSO` | `IC293__1_MMStack_Pos62-DMSO.ome-cell3-cropped.tif` | 44×658×508 |
| `Pos62-DMSO` | `IC293__1_MMStack_Pos62-DMSO.ome-cropped.tif` | 40×430×580 |
| `Pos63-DMSO` | `IC293__1_MMStack_Pos63-DMSO.ome-cropped.tif` | 35×614×696 |
| `Pos64-DMSO` | `IC293__1_MMStack_Pos64-DMSO.ome-cropped.tif` | 97×602×576 |
| `Pos66-DMSO` | `IC293__1_MMStack_Pos66-DMSO.ome-cropped.tif` | 50×614×704 |
| `Pos68-DMSO` | `IC293__1_MMStack_Pos68-DMSO.ome-cropped.tif` | 97×478×686 |
| `Pos70-DMSO` | `IC293__1_MMStack_Pos70-DMSO.ome-cropped.tif` | 97×532×520 |
| `Pos71-DMSO` | `IC293__1_MMStack_Pos71-DMSO.ome-cropped.tif` | 97×624×540 |
