# Off-market presentation — Member B

A three-slide, approximately 90-second segment covering the approach, measured
results, and final off-market submission, with two data-based graphs.
The revised first slide explains a five-step working process: select allowed
inputs, engineer features, fit category encoders within the training fold,
learn log prices with boosting trees, and convert predictions back to dollars.
The next slides explain monthly validation, final refitting, and CSV checks.

- [Slides PDF](off_market_90_second_slides.pdf)
- [Speaker script PDF](off_market_90_second_script.pdf)
- [Editable speaker script](off_market_90_second_script.docx)
- [Plain-text speaker script](off_market_90_second_script.txt)

The script contains 190 written words. Rehearse at roughly 130 words per minute,
with brief slide-transition pauses. Your 90-second segment leaves 3 minutes
30 seconds for the rest of the team's five-minute video.

The final original-data validation measured **8.53% MdAPE** and **11.00% MAPE**
across **10,069 held-out homes**, December 2025-May 2026. Reference baselines use
the same validation homes and original sale-price labels from the earlier run.
The final fit used **150,000 original training sales** and produced **3,974**
checked off-market predictions. Hidden-test accuracy remains unknown.
