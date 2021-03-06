# -*- coding:utf-8 -*-
# 知能機械情報学レポート課題１
# 実行環境：Julia v0.6.2

using Plots, RandomNumbers, StatsBase

# 検証用乱数生成器
rnd = MersenneTwister(111)

# define constant
const theta = 0.
const imgdim = 5  # 画像の一辺画素
const itr = 10

#======================
= define functions... =
======================#
#活性化関数（符号関数）
function sgn(x::AbstractFloat)
    if x >= 0
        return 1
    else
        return -1
    end
end

# ネットワークエネルギー関数
lyapnov(w::AbstractMatrix,xr::AbstractVector) = xr' * w * xr + sum(theta * xr)

# ノイズ生成関数
function pat_noise_mk(pat_ar::AbstractArray, ratio::AbstractFloat)
    pat_ar_ = deepcopy(pat_ar)
    for pat in pat_ar_
        pick = cld(size(pat)[1]*ratio,1)
        er_cf = sample(1:size(pat)[1],Int(pick), replace=false)
        for i in er_cf
            pat[i] = -pat[i]
        end
    end
    return pat_ar_
end

#==========================
= define execute function =
==========================#
# 訓練パターン生成関数
function make_train_data(num::Int, dim::Int)
    train = [sgn.(rand(rnd, dim^2)-0.5) for i in 1:num]
    return train
end

# 学習
function hnn_learning(pat_ar::AbstractVector)
    neusize = imgdim^2
    weight = zeros(neusize, neusize)  #重み行列の初期化
    Q = size(pat_ar)[1]
    weight = sum(pat*transpose(pat) for pat in pat_ar)/Q
    for i in 1:neusize
        weight[i,i] = 0.
    end
    return weight
end

# 自己想起関数
function hnn_predict(weight::AbstractMatrix, test::AbstractVector, loop=1000)
    eng = zeros(size(test)[1])
    xr = deepcopy(test)
    for j in 1:size(test)[1]
        brk_cnt = 0
        eng[j] = lyapnov(weight, test[j])
        for i in 1:loop
            ind = rand(1:size(test[1])[1],1)   # ニューロンを一つランダムサンプリング
            xr_ = weight * xr[j] - theta
            xr[j][ind] = sgn.(xr_)[ind]
            eng_ = lyapnov(weight, xr[j])
            # 収束判定
            if eng_ == eng[j]
                brk_cnt += 1
                if brk_cnt >= 200
                    break
                end
            else
                brk_cnt = 0
            end
            eng[j] = eng_
        end
    end
    return xr, eng
end

# 画像出力関数
function pat_img(pat_ar::AbstractArray, dim::Int, title::String="title")
    i = collect(1:1:dim)
    j = collect(1:1:dim)
    pat_ar = [reshape(pat, (dim,dim)) for pat in pat_ar]
    imgs = [heatmap(ppat, bar_width=false, title=title) for ppat in pat_ar]
    return imgs #パターン画像のリストを戻す
end
pat_img(weight::AbstractMatrix) = heatmap(weight, bar_width=false, size=(500,500), title="weight")
out_plot(imt, imi, imo, ind::Int=1) = plot(imt[ind],imi[ind], imo[ind], layout=(1,3),size=(600,200))

function HopfieldNet(pats::AbstractVector, noise_ratio::AbstractFloat)
    weight = hnn_learning(pats)
    pats_err = pat_noise_mk(pats, noise_ratio)
    pats_pre, eng = hnn_predict(weight, pats_err)
    return weight ,eng, pats_err, pats_pre
end

function hnn_eval(pats, noise_ratio=0.1)
    # 適当な初期化（しないとfor内のローカル変数として処理されるっぽい）
    pats_in = deepcopy(pats)
    pats_out = deepcopy(pats)
    eng = zeros(size(pats)[1])
    # 正誤率初期化
    tf_cnt = zeros(size(pats)[1])
    anti = zeros(size(pats)[1])
    weight = zeros(imgdim^2, imgdim^2)

    for i in 1:itr
        weight ,eng, pats_in, pats_out = HopfieldNet(pats, noise_ratio)
        for i in 1:size(pats)[1]
            if pats[i] == pats_out[i]
                tf_cnt[i] +=1
            end
            if pats[i] == -pats_out[i]
                anti[i] += 1
            end
        end
    end

    tf_cnt = tf_cnt / itr
    anti = anti / itr

    imgs_train = pat_img(pats, imgdim, "train_data")
    imgs_in = pat_img(pats_in, imgdim, "test_data")
    imgs_out = pat_img(pats_out, imgdim, "output")
    return weight, imgs_train, imgs_in, imgs_out, tf_cnt, anti
end

#=======================
= define main function =
=======================#

function remember()
    println("remember initial image...")
    weight = zeros(imgdim^2, imgdim^2)
    noises = collect(0.04:0.04:0.2)
    pat = make_train_data(1,imgdim)
    imt = []
    imi = []
    imo = []

    for (nz,i) in zip(noises, 1:size(noises)[1])
        weight, imgs_train, imgs_in, imgs_out, tf_cnt, anti = hnn_eval(pat, nz)
        push!(imt, imgs_train)
        push!(imi, imgs_in)
        push!(imo, imgs_out)
    end
    return imt, imi, imo, weight
end

function many_pats()
    println("memorize many patterns...")
    pat_size = [1,2,3,4,5,6,7]
    tfs = zeros(size(pat_size)[1])
    main_itr = 100
    for j in 1:main_itr
        for i in pat_size
            # 評価画像の生成
            pats = make_train_data(i, imgdim)
            weight, imgs_train, imgs_in, imgs_out, tf_cnt, anti = hnn_eval(pats)
            tfs[i] += mean(tf_cnt)   #平均正解率
        end
    end
    tfs = tfs/main_itr * 100
    return tfs
end

function many_noise()
    println("memorize kinds of noise pattern...")
    noises = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    tfs2 = zeros(11)
    tfs4 = zeros(11)
    anti2 = zeros(11)
    anti4 = zeros(11)
    main_itr = 100

    for j in 1:main_itr
        pats2 = make_train_data(2, imgdim)
        pats4 = make_train_data(4, imgdim)
        for nz in 0:10
            weight, imgs_train, imgs_in, imgs_out, tf_cnt, anti = hnn_eval(pats2, nz/10)
            tfs2[nz+1] += mean(tf_cnt)
            anti2[nz+1] += mean(anti)
            weight, imgs_train, imgs_in, imgs_out, tf_cnt, anti = hnn_eval(pats4, nz/10)
            tfs4[nz+1] += mean(tf_cnt)
            anti4[nz+1] += mean(anti)
        end
    end
    tfs2 = tfs2/main_itr * 100
    tfs4 = tfs4/main_itr * 100
    anti2 = anti2/main_itr * 100
    anti4 = anti4/main_itr * 100
    return tfs2, tfs4, noises, anti2, anti4
end

# main routine
imt, imi, imo, weight = @time remember()
tfs = @time many_pats()
tfs2, tfs4, noises, anti2, anti4 = @time many_noise()
out_plot(imt[1], imi[1], imo[1])
savefig("out1.png")
out_plot(imt[2], imi[2], imo[2])
savefig("out2.png")
out_plot(imt[3], imi[3], imo[3])
savefig("out3.png")
out_plot(imt[4], imi[4], imo[4])
savefig("out4.png")
out_plot(imt[5], imi[5], imo[5])
savefig("out5.png")
pat_img(weight)
savefig("weight.png")
plot(tfs,xlabel="No. of patterns", ylabel="accuracy[%]",ylims=[0,100])
savefig("out6.png")
plot(noises,tfs2, label="two type memorized",xlabel="noise ratio", ylabel="accuracy[%]")
plot!(noises,tfs4, label="four type memorized")
savefig("out7.png")
plot(noises,tfs2, label="2-true pattern",xlabel="noise ratio", ylabel="accuracy[%]")
plot!(noises,tfs4, label="4-true pattern")
plot!(noises,anti2, label="2-anti pattern")
plot!(noises,anti4, label="4-anti pattern")
savefig("out8.png")
