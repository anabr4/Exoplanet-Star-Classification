# Exoplanet-Star-Classification
In this project, we will analyse the flux data from thousands of stars via deep learning methods in order to classify them as either exoplanet-stars or non-exoplanet-stars. We will train one model on the training dataset in order to make predictions on the test dataset.

An exoplanet, or extrasolar planet, is the one that orbits a star outside our solar system. They give us information of the formation, composition, and diversity of planetary systems beyond the Solar system and they can vary in size, composition and orbital characteristics. Some of the types are terrestrial, super-earths, gas giants or neptune-like exoplanets.

## 1. The Data: Star Light Intensities vs Time
The [data](https://www.kaggle.com/keplersmachines/kepler-labelled-time-series-data) has been cleaned and obtained from observations made by the NASA Kepler space telescope. It describe the change in flux (light intensity)(units not mentioned) of several thousand stars, with binary label 2 or 1, indicating the presence of at least one exoplanet in orbit or its absence, respectively. The transit method, used in this dataset, is an indirect method for detecting exoplanets, which consists in observing periodic dimming of a star's light intensity as a planet passes in front of it, as shown in the picture below.
![image](https://github.com/user-attachments/assets/84389833-a3bb-481c-a84f-f36f443172ea)
(credit: NASA Ames)

Even if stars exhibit dimming in their intensity period, further study is required to confirm its existence, e.g. employing satellites capturing different wavelengths that provide additional data to ensure the results obteined with the transit method.

Data provided is already divided into Training and Testing data.

Executing the data visualization 
