<div align="center" markdown>
<img src="https://user-images.githubusercontent.com/106374579/186665737-ec3da9cc-193f-43ee-85db-a6f802b2dfe4.png"/>

# Export images to Supervisely Format

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Use">How To Use</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/export-to-supervisely-format)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/export-to-supervisely-format)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/export-to-supervisely-format.png)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/export-to-supervisely-format.png)](https://supervise.ly)

</div>

# Overview

ℹ️ Starting from version 2.7.7 the application will save images metadata in JSON format to `meta` directory in each dataset.

Download images project or dataset in [Supervisely JSON format](https://docs.supervise.ly/data-organization/00_ann_format_navi). It is possible to download both images and annotations or only annotations.

# How To Use

**Step 1**: Add app to your team from [Ecosystem](https://ecosystem.supervise.ly/apps/export-to-supervisely-format) if it is not there

**Step 2**: Open context menu of images project (or images dataset) -> `Run App` -> `Download via app` -> `Export to Supervisely format`

<img src="https://i.imgur.com/6JNfu3g.png" width="600px"/>

**Step 3**: Define export settings in modal window

<img src="https://i.imgur.com/jXSSOTW.png" width="600px">

**Step 4**: Result archive will be available for download in `Tasks` list (image below) or from `Team Files` (path format is the following `Team Files`->`Export-to-Supervisely`->`<task_id>_<projectId>_<projectName>.tar`)

<img src="https://i.imgur.com/QjFHRtx.png">
