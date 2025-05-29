Bu projede, MSTAR veri setindeki 4 sınıf üzerinde görüntü işleme filtrelerini inceleyip bu veri seti için bir sınıflandırma modeli oluşturdum. Amacım, havadan çekilmiş askeri araçların Yapay Açıklıklı Radar (SAR) görüntülerine uygulanan filtrelerin, bu görüntüler üzerinde eğitilen sınıflandırma modelinin performansına etkisini gözlemlemek. Hem filtreli hem de filtresiz görüntülerle yapılan eğitim sonuçlarını karşılaştırarak, filtrelerin model performansına olan etkisini analiz etmeye çalıştım.

MSTAR veri setini [buradan](https://www.kaggle.com/datasets/atreyamajumdar/mstar-dataset-8-classes) indirdikten sonra, Python yardımıyla kopya olan fotoğrafları temizledim. Veri setinden 4 sınıf seçerek projede kullanmak üzere ayırdım. Bu seçilen sınıflardaki fotoğraflara EAW (Edge-Aware Wavelet), Lee, BM3D (Block-Matching and 3D Filtering) ve Median filtrelerini uyguladım. Filtrelerin parametreleriyle oynayarak en optimum sonuçları bulmaya çalıştım.

Filtrelerin sonuçlarını karşılaştırdım.
![comp](https://github.com/harunrk/mstar_classifier/blob/main/Comparison%20of%20filters.png) 

En iyi sonuçları Medyan ve BM3D filtreleri verdi. Bu filtrelerin üzerine, nesnelerin kenarlarını daha belirgin hale getirmek için keskinleştirme filtresi uyguladım. Keskinleştirme filtresi uyguladığımda benek gürültüsünün arttığını gözlemledim, ancak kenarların daha belirginleştiğini gördüm. 

![comp2](https://github.com/harunrk/mstar_classifier/blob/main/sharpened_images.png) 

Bu işlemleri yaparken yazdığım Python kodlarına filterin.zip klasörü üzerinden ulaşabilirsiniz.

Görüntü işleme adımlarından sonra LeNet-5 mimarisine benzer bir model oluşturdum ve görsellerin önişlemesini sağlamak için uygun kodu yazdım. 

![LeNet-5](https://github.com/harunrk/mstar_classifier/blob/main/lenet.png)

Modelimi ilk olarak filtre uygulanmamış görüntüler ile eğittim ve doğruluk oranını karmaşıklık matrisi ile test ettim. Daha sonra aynı işlemleri BM3D+Keskinleştirme ve Median+Keskinleştirme filtreleri uygulanmış görüntüler ile de gerçekleştirdim.

50 epoch ve 32 batch size kullanarak eğittiğim üç modelin karmaşıklık matrisleri ve doğruluk oranları aşağıdaki gibidir:

   Filtre uygulanmamış görseller ile oluşturulan model (Test accuracy: 0.9779)
   
   ![Original](https://github.com/harunrk/mstar_classifier/blob/main/original%20conf.png)

   Median+Keskinleştirme filtresi uygulanmış görseller ile oluşturulan model (Test accuracy: 0.9867)
   
   ![Median](https://github.com/harunrk/mstar_classifier/blob/main/sMedian_conf.png)

   BM3D+Keskinleştirme filtresi uygulanmış görseller ile oluşturulan model (Test accuracy: 0.9889)
   
   ![BM3D](https://github.com/harunrk/mstar_classifier/blob/main/sharpened_bm3d_conf.png)


## NOT
Model eğitilirken sürecin daha hızlı ilerleyebilmesi için label encoder işlemi uygulandı dolayısıyla 0: 2S1, 1: BTR_60, 2: T62 ve 3: ZIL131 sınıflarını temsil etmektedir
