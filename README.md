# Automatically classify consumer goods [OC-P6]

## **Problem to solve**

You are a Data Scientist within the **Marketplace** company, which wishes to launch an e-commerce marketplace.

In the Marketplace, sellers offer items to buyers by posting a photo and description.

Currently, the assignment of an item's category is done manually by sellers and is therefore unreliable. In addition, the volume of articles is currently very small.

To make the user experience for sellers (making it easier to put new items online) and buyers (making it easier to find products) as smooth as possible and with a view to scaling up, it becomes necessary to 'automate this task.

## **Your mission**

**Study** the feasibility of an engine for classifying articles into different categories, with a sufficient level of precision.

Your mission is to to carry out a first feasibility study of an article classification engine based on an image and a description for the automation of the attribution of the article category.

1. Analyze the dataset by performing preprocessing of product images and descriptions, dimension reduction, and then clustering.. 
2. The results of the clustering will be presented in the form of a two-dimensional representation to be determined, which will illustrate that the characteristics extracted allow products of the same category to be grouped together.

### **Considerations**

- The graphical representation will help you convince Linda that this modeling approach will allow you to group products of the same category.

- In order to extract the features, implement at least a SIFT / ORB / SURF type algorithm. A CNN Transfer Learning type algorithm can optionally be used as a supplement, if it can shed additional light on the demonstration.

- Please note, it is not needed a classification engine at this point, but a feasibility study!

## **The data**

For this mission, the **Marketplace** has provided images with an  database containing information about the products the following folders.

- datasets/flipkart_com-ecommerce_sample_1050.csv
- images/Flipkart

The complete data is available on the following link

- [**E-Commerce data**](https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/Parcours_data_scientist/Projet+-+Textimage+DAS+V2/Dataset+projet+pre%CC%81traitement+textes+images.zip)

## **Repository file structure**

- cleaning_notebook.ipynb: Cleaning notebook
- modeling_notebook.ipynb: Notebook with predictions
- datasets: datasets of the project
- images: Images and graphs of the project
- supports: Folder with documents to support the work done
    - Project 5 presentation: Project presentation in French

### **Final note**

- The notebook is optimized to be used with **Jupyter lab**.