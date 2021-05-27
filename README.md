<div align="center" markdown>
<img src="https://i.imgur.com/"/>

# Download images and annotations

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Use">How To Use</a>
</p>


[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/download_as_supervisely)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/download_as_supervisely)
[![views](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/download_as_supervisely&counter=views&label=views)](https://supervise.ly)
[![used by teams](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/download_as_supervisely&counter=downloads&label=used%20by%20teams)](https://supervise.ly)
[![runs](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/download_as_supervisely&counter=runs&label=runs)](https://supervise.ly)

</div>

# Overview

This application allows to download both images and their annotations (only annotations in `*.json` fromat or ). The result  will be saved in a new project. 
Export prepares downloadable .tar archive, that contains:
- only annotations in `*.json` fromat:
  - annotations in [Supervisely JSON format](https://docs.supervise.ly/data-organization/00_ann_format_navi)

- whole project as `*.json + images`:
  - original images
  - annotations in [Supervisely JSON format](https://docs.supervise.ly/data-organization/00_ann_format_navi)

# How To Use
**Step 1**: Add app to your team from [Ecosystem](https://app.supervise.ly/apps/ecosystem/export-as-masks) if it is not there

    
**Step 2**: Open context menu of images project -> `Run App` -> `Export` -> `Downloas as images` 

<img src="https://i.imgur.com/GRWZbuU.png"/>

**Step 3**: Define export settings in modal window

<img src="https://i.imgur.com/0ZhbOjx.png">

**Step 4**: Result archive will be available for download in `Tasks` list (image below) or from `Team Files` (path format is the following `Team Files`->`Download-data`->`<task_id>_<projectId>_<projectName>.tar`)

<img src="https://i.imgur.com/I0umhsL.png">
