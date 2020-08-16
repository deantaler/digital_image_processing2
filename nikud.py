import sys
import cv2

# Loading letters with punctuations, detecting each shape and compare it with a punctuation in the background file. " \
# Return the image without punctuations."
# @argv[1] = 'nikud' image
# @argv[2] = 'words' image

# Changes img to binary color and find contour of 'nikud' shapes
path_nik = sys.argv[1]
nik = cv2.imread(path_nik, cv2.IMREAD_GRAYSCALE)
_, threshold_nik = cv2.threshold(nik, 230, 255, cv2.THRESH_BINARY)
contours_nik, _ = cv2.findContours(threshold_nik, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

path_word = sys.argv[2]
words = cv2.imread(path_word, cv2.IMREAD_GRAYSCALE)
_, threshold_words = cv2.threshold(words, 230, 255, cv2.THRESH_BINARY)
contours_words, _ = cv2.findContours(threshold_words, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


# Gets two contours arrays and coloring in white each shape in word's file that appear in nikud's file
def compare(nikud, word):
    # Parameter 'i' will indicate if the current shape is the frame
    i = 0

    # Runs over shapes from 'words' file
    for letter in word:
        approx_words = cv2.approxPolyDP(letter, 0.001 * cv2.arcLength(letter, True), True)
        # letter_area = cv2.contourArea(approx_words, True)

        # Runs over shapes from 'nikud' file
        for shape in nikud:
            approx_nikud = cv2.approxPolyDP(shape, 0.001 * cv2.arcLength(shape, True), True)
            # nik_area = cv2.contourArea(approx_nikud, True)

            if cv2.matchShapes(approx_words, approx_nikud, cv2.CONTOURS_MATCH_I3, 0) <= 0.002 and i > 0:
                # if letter_area == nik_area and i > 0:
                cv2.fillConvexPoly(words, approx_words, 255)

        i = i+1

    cv2.imshow("words", words)


compare(contours_nik, contours_words)

cv2.waitKey(0)
cv2.destroyAllWindows()
