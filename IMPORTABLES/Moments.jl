module Moments

using Random, CSV, DataFrames, LinearAlgebra, Printf, Statistics
import Statistics: mean

export firstMoment, secondMoment, thirdMoment, fourthMoment, fifthMoment, sixthMoment, seventhMoment, eighthMoment, getFiles, allMoments, oneFileMoments, allMomentsMAE

function getFiles(GMM_NAME, INFER_NAME, OBSERVATION_NAME)
    mat_gmm = Matrix(CSV.read(GMM_NAME, DataFrame))
    mat_inf = Matrix(CSV.read(INFER_NAME, DataFrame))
    mat_obs = Matrix(CSV.read(OBSERVATION_NAME, DataFrame))
    return mat_gmm, mat_inf, mat_obs
end

function getErr(x, y)
    return norm(x.-y)/sqrt(length(x))
end

function getMAE(x, y)
    return sum(abs.(x .- y)) / length(x)
end

function firstMoment(mat)
    return vec(mean(mat, dims=1))
end 

function secondMoment(mat)
    return (mat'*mat)/size(mat,1)
end 

function thirdMoment(mat)
    N, D = size(mat)
    T = zeros(D, D, D)
    for n in 1:N
        x = mat[n, :]
        for i in 1:D, j in 1:D, k in 1:D
            T[i,j,k] += x[i]*x[j]*x[k]
        end
    end
    return T./N
end

function fourthMoment(mat)
    N, D = size(mat)
    T = zeros(D, D, D, D)
    for n in 1:N
        x = mat[n, :]
        for i in 1:D, j in 1:D, k in 1:D, l in 1:D
            T[i,j,k,l] += x[i]*x[j]*x[k]*x[l]
        end
    end
    return T./N
end

function fifthMoment(mat)
    N, D = size(mat)
    T = zeros(D, D, D, D, D)
    for n in 1:N
        x = mat[n, :]
        for i in 1:D, j in 1:D, k in 1:D, l in 1:D, m in 1:D
            T[i,j,k,l,m] += x[i]*x[j]*x[k]*x[l]*x[m]
        end
    end
    return T ./ N
end

function sixthMoment(mat)
    N, D = size(mat)
    T = zeros(D, D, D, D, D, D)
    for n in 1:N
        x = mat[n, :]
        for i in 1:D, j in 1:D, k in 1:D, l in 1:D, m in 1:D, o in 1:D
            T[i,j,k,l,m,o] += x[i]*x[j]*x[k]*x[l]*x[m]*x[o]
        end
    end
    return T ./ N
end


function seventhMoment(mat)
    N, D = size(mat)
    T = zeros(D, D, D, D, D, D, D)
    for n in 1:N
        x = mat[n, :]
        for i in 1:D, j in 1:D, k in 1:D, l in 1:D, m in 1:D, o in 1:D, p in 1:D
            T[i,j,k,l,m,o,p] += x[i]*x[j]*x[k]*x[l]*x[m]*x[o]*x[p]
        end
    end
    return T ./ N
end


function eighthMoment(mat)
    N, D = size(mat)
    T = zeros(D, D, D, D, D, D, D, D)
    for n in 1:N
        x = mat[n, :]
        for i in 1:D, j in 1:D, k in 1:D, l in 1:D, m in 1:D, o in 1:D, p in 1:D, q in 1:D
            T[i,j,k,l,m,o,p,q] += x[i]*x[j]*x[k]*x[l]*x[m]*x[o]*x[p]*x[q]
        end
    end
    return T ./ N
end


function allMoments(gmm, inf, obs; outname="moments_summary.txt", minimal=false)
    open(outname, "w") do io
        println(io, "FILE HEADER: ", outname)
        # --- first moment ---
        gmm_1 = firstMoment(gmm)
        inf_1 = firstMoment(inf)
        obs_1 = firstMoment(obs)
        err_gmm1 = getErr(gmm_1, obs_1)
        err_inf1 = getErr(inf_1, obs_1)

        println(io, "===== FIRST MOMENT =====")
        println(io, "err_gmm1 = ", err_gmm1, " | err_inf1 = ", err_inf1)
        if !minimal
            println(io, "gmm_1 = ", gmm_1)
            println(io, "inf_1 = ", inf_1)
            println(io, "obs_1 = ", obs_1, "\n")
        end

        # --- second moment ---
        gmm_2 = secondMoment(gmm)
        inf_2 = secondMoment(inf)
        obs_2 = secondMoment(obs)
        err_gmm2 = getErr(gmm_2, obs_2)
        err_inf2 = getErr(inf_2, obs_2)

        println(io, "===== SECOND MOMENT =====")
        println(io, "err_gmm2 = ", err_gmm2, " | err_inf2 = ", err_inf2)
        if !minimal
            println(io, "gmm_2 = ", gmm_2)
            println(io, "inf_2 = ", inf_2)
            println(io, "obs_2 = ", obs_2, "\n")
        end

        # --- third moment ---
        gmm_3 = thirdMoment(gmm)
        inf_3 = thirdMoment(inf)
        obs_3 = thirdMoment(obs)
        err_gmm3 = getErr(gmm_3, obs_3)
        err_inf3 = getErr(inf_3, obs_3)

        println(io, "===== THIRD MOMENT =====")
        println(io, "err_gmm3 = ", err_gmm3, " | err_inf3 = ", err_inf3)
        if !minimal
            println(io, "gmm_3 = ", gmm_3)
            println(io, "inf_3 = ", inf_3)
            println(io, "obs_3 = ", obs_3, "\n")
        end
        
        # --- fourth moment ---
        gmm_4 = fourthMoment(gmm)
        inf_4 = fourthMoment(inf)
        obs_4 = fourthMoment(obs)
        err_gmm4 = getErr(gmm_4, obs_4)
        err_inf4 = getErr(inf_4, obs_4)

        println(io, "===== FOURTH MOMENT =====")
        println(io, "err_gmm4 = ", err_gmm4, " | err_inf4 = ", err_inf4)
        if !minimal
            println(io, "gmm_4 = ", gmm_4)
            println(io, "inf_4 = ", inf_4)
            println(io, "obs_4 = ", obs_4, "\n")
        end
        """
        # --- fifth moment ---
        println("going fifth")
        gmm_5 = fifthMoment(gmm)
        inf_5 = fifthMoment(inf)
        obs_5 = fifthMoment(obs)
        err_gmm5 = getErr(gmm_5, obs_5)
        err_inf5 = getErr(inf_5, obs_5)

        println(io, "===== FIFTH MOMENT =====")
        println(io, "err_gmm5 = ", err_gmm5, " | err_inf5 = ", err_inf5)
        if !minimal
            println(io, "gmm_5 = ", gmm_5)
            println(io, "inf_5 = ", inf_5)
            println(io, "obs_5 = ", obs_5, "\n")
        end

        # --- sixth moment ---
        gmm_6 = sixthMoment(gmm)
        inf_6 = sixthMoment(inf)
        obs_6 = sixthMoment(obs)
        err_gmm6 = getErr(gmm_6, obs_6)
        err_inf6 = getErr(inf_6, obs_6)

        println(io, "===== SIXTH MOMENT =====")
        println(io, "err_gmm6 = ", err_gmm6, " | err_inf6 = ", err_inf6)
        if !minimal
            println(io, "gmm_6 = ", gmm_6)
            println(io, "inf_6 = ", inf_6)
            println(io, "obs_6 = ", obs_6, "\n")
        end

        # --- seventh moment ---
        gmm_7 = seventhMoment(gmm)
        inf_7 = seventhMoment(inf)
        obs_7 = seventhMoment(obs)
        err_gmm7 = getErr(gmm_7, obs_7)
        err_inf7 = getErr(inf_7, obs_7)

        println(io, "===== SEVENTH MOMENT =====")
        println(io, "err_gmm7 = ", err_gmm7, " | err_inf7 = ", err_inf7)
        if !minimal
            println(io, "gmm_7 = ", gmm_7)
            println(io, "inf_7 = ", inf_7)
            println(io, "obs_7 = ", obs_7, "\n")
        end

        # --- eighth moment ---
        gmm_8 = eighthMoment(gmm)
        inf_8 = eighthMoment(inf)
        obs_8 = eighthMoment(obs)
        err_gmm8 = getErr(gmm_8, obs_8)
        err_inf8 = getErr(inf_8, obs_8)

        println(io, "===== EIGHTH MOMENT =====")
        println(io, "err_gmm8 = ", err_gmm8, " | err_inf8 = ", err_inf8)
        if !minimal
            println(io, "gmm_8 = ", gmm_8)
            println(io, "inf_8 = ", inf_8)
            println(io, "obs_8 = ", obs_8, "\n")
        end
        """

    end

    println("Saved all moments and errors (full precision) to text file.")
end



function allMomentsMAE(gmm, inf, obs; outname="moments_summary.txt", minimal=false)
    open(outname, "w") do io
        println(io, "FILE HEADER: ", outname)
        # --- first moment ---
        gmm_1 = firstMoment(gmm)
        inf_1 = firstMoment(inf)
        obs_1 = firstMoment(obs)
        err_gmm1 = getMAE(gmm_1, obs_1)
        err_inf1 = getMAE(inf_1, obs_1)

        println(io, "===== FIRST MOMENT =====")
        println(io, "err_gmm1 = ", err_gmm1, " | err_inf1 = ", err_inf1)
        if !minimal
            println(io, "gmm_1 = ", gmm_1)
            println(io, "inf_1 = ", inf_1)
            println(io, "obs_1 = ", obs_1, "\n")
        end

        # --- second moment ---
        gmm_2 = secondMoment(gmm)
        inf_2 = secondMoment(inf)
        obs_2 = secondMoment(obs)
        err_gmm2 = getMAE(gmm_2, obs_2)
        err_inf2 = getMAE(inf_2, obs_2)

        println(io, "===== SECOND MOMENT =====")
        println(io, "err_gmm2 = ", err_gmm2, " | err_inf2 = ", err_inf2)
        if !minimal
            println(io, "gmm_2 = ", gmm_2)
            println(io, "inf_2 = ", inf_2)
            println(io, "obs_2 = ", obs_2, "\n")
        end

        # --- third moment ---
        gmm_3 = thirdMoment(gmm)
        inf_3 = thirdMoment(inf)
        obs_3 = thirdMoment(obs)
        err_gmm3 = getMAE(gmm_3, obs_3)
        err_inf3 = getMAE(inf_3, obs_3)

        println(io, "===== THIRD MOMENT =====")
        println(io, "err_gmm3 = ", err_gmm3, " | err_inf3 = ", err_inf3)
        if !minimal
            println(io, "gmm_3 = ", gmm_3)
            println(io, "inf_3 = ", inf_3)
            println(io, "obs_3 = ", obs_3, "\n")
        end
        
        # --- fourth moment ---
        gmm_4 = fourthMoment(gmm)
        inf_4 = fourthMoment(inf)
        obs_4 = fourthMoment(obs)
        err_gmm4 = getMAE(gmm_4, obs_4)
        err_inf4 = getMAE(inf_4, obs_4)

        println(io, "===== FOURTH MOMENT =====")
        println(io, "err_gmm4 = ", err_gmm4, " | err_inf4 = ", err_inf4)
        if !minimal
            println(io, "gmm_4 = ", gmm_4)
            println(io, "inf_4 = ", inf_4)
            println(io, "obs_4 = ", obs_4, "\n")
        end
    end

    println("Saved all moments and errors (full precision) to text file.")
end




function oneFileMoments(obs; outname="moments_obs.txt")
    open(outname, "w") do io
        println(io, "FILE HEADER: ", outname)
        obs_1 = firstMoment(obs)
        println(io, "obs_1 = ", obs_1, "\n")

        obs_2 = secondMoment(obs)
        println(io, "obs_2 = ", obs_2, "\n")

        obs_3 = thirdMoment(obs)
        println(io, "obs_3 = ", obs_3, "\n")
        
        obs_4 = fourthMoment(obs)
        println(io, "obs_4 = ", obs_4, "\n")
    end

    println("Saved all moments and errors (full precision) to text file.")
end


end
# include("../../../IMPORTABLES/Moments.jl")
# using .Moments
# gmm, inf, obs = getFiles("gmm_samples.csv", "inferred_samples.csv", "observed.csv")
# allMoments(gmm, inf, obs)
