import numpy as np
rng = np.random.default_rng(42)
subjects = ["Maths", "English", "Kiswahili", "Science", "SST"]
scores = rng.integers(30, 100, size=(30, 5)) # 30 students x 5 subjects

#1. Print the shape, dtype and overall mean of scores.
print(scores.shape)
print(scores.dtype)
print(scores.mean().round(2))

#2.Find the mean score per subject (axis=0) and the total score per student (axis=1).
print(scores.mean(axis=0).round(2))
print(scores.mean(axis=1))
#3.Find the index of the best student (highest total) and the subject with the highest mean
print(scores.argmax())
print(scores.mean(axis=0).argmax().round(2))

#4.How many students scored above 80 in Maths? What percentage of the class is that?
maths=scores[ :,0]
best_maths=maths>80
print(best_maths.sum())
above=best_maths.sum()
percentage=above/30 *100
print(percentage.round(2))


#5.Standardise each subject using z-scores: (x - mean) / std. Check that each column now has mean close to 0.

z = (scores - scores.mean(axis=0)) / scores.std(axis=0)

print(z.mean(axis=0).round(6)) 
# all about 0

#6. Add a 6th column holding each student's average. Then use np.where to give grades: A (75+), B(60-74), C (50-59), D (below 50). Count each grade with np.unique.

students_avg=scores.mean(axis=1)

scores6 = np.column_stack([scores, students_avg])

grade = np.where(students_avg >= 75, "A",

              np.where(students_avg >= 60, "B",

                np.where(students_avg >= 50, "C", "D")))

labels, counts = np.unique(grade, return_counts=True)
labels, counts = np.unique(grade, return_counts=True)
labels, counts = np.unique(grade, return_counts=True)
print(dict(zip(labels, counts)))


#7. Find the correlation between Maths and Science using np.corrcoef. What does it tell you?

r = np.corrcoef(scores[:, 0], scores[:, 3])[0, 1]
print("Maths vs Science correlation:", round(r, 2))
# Close to 0 
